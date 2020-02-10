from paddle import fluid
from paddle.fluid import layers
from paddle.fluid import dygraph
import numpy as np

if __name__ != "__main__":
    from model.dy_mobilenetV2 import MobileNetV2

_configuration = {
    'vocab_size': 7946, # 字典大小
    'embedding_size': 256,
    'sentence_length': 39, # 训练用句子的最长长度(包括<start>和<end>)
    'padding_idx': 0,
    'start_idx': 2, # `<start>`` 的id
    'encoder_dim': 7 * 7,
    'encoder_channel': 1280, 
    'hidden_dim': 1024,
    'enhancement':{
        'doubly_stochastic_attention': False, # 见论文(4.2.1)
        'gating_scalar': False, # 见论文(4.2.1)
        'dropout': False
    },
    'use_cudnn': False,
    'infer_max_length': 50,
}

class ImageCaptioning(dygraph.layers.Layer):

    def __init__(self, name_scope):
        super(ImageCaptioning, self).__init__(name_scope)

        self._weight_count = 0
        self._bias_count = 0

        self.encoder = MobileNetV2('encoder')
        self._batch_norm = dygraph.nn.BatchNorm(self.full_name(), _configuration['encoder_channel'])
        self._embedding = dygraph.nn.Embedding(self.full_name(), 
            (_configuration['vocab_size'], _configuration['embedding_size']),
            is_sparse=True,
            padding_idx=_configuration['padding_idx'])

        self._initial_lstm_h = dygraph.nn.FC(self.full_name(), _configuration['hidden_dim'], act='tanh')
        self._initial_lstm_c = dygraph.nn.FC(self.full_name(), _configuration['hidden_dim'], act='tanh')

        self._att_hidden_proj = dygraph.nn.FC(self.full_name(), _configuration['encoder_channel'])
        if _configuration['enhancement']['gating_scalar']:
            self._selector_hidden_proj = dygraph.nn.FC(self.full_name(), 1)

        self._lstm_fc = dygraph.nn.FC(self.full_name(), 4 * _configuration['hidden_dim'])

        self._decode_hidden_proj = dygraph.nn.FC(self.full_name(), _configuration['embedding_size'])
        self._decode_output = dygraph.nn.FC(self.full_name(), _configuration['vocab_size'])

    # create_parameter 在__init__中调用会导致接下来matmul时出错
    # 似乎必须要在_build_once里调用才正常
    # 弄得我还以为得用append_op
    # 文档里又不说......
    # 真的服了......
    def _build_once(self, input, sentence=None):
        # 根据文档，所有有参数的东西都得是dynamic下的
        # layers.create_parameter() 不能用
        self.feature_proj_w = self.createParam([_configuration['encoder_channel']] * 2)
        self.att_w = self.createParam([_configuration['encoder_channel'], 1])
        self.decode_ctx2out = self.createParam([_configuration['encoder_channel'], _configuration['embedding_size']])
        # Debug
        # self.all_ones = self.createParam([3, 1, 4], initializer=fluid.initializer.Constant(1))

    def forward(self, image, sentence=None):
        print(image.dtype)
        feature = self.encoder(image)
        feature = self._batch_norm(feature)
        feature = layers.reshape(feature, 
            [-1, _configuration['encoder_dim'], _configuration['encoder_channel']])

        feature_proj = self._feature_proj(feature)

        c, h = self._get_initial_lstm(feature)

        if sentence is not None: # train
            sentence_in = sentence[:, :-1]
            sentence_out = sentence[:, 1:]
            pad = layers.fill_constant(sentence_out.shape, dtype='int64', value=_configuration['padding_idx'])
            padding_mask = layers.cast(layers.logical_not(layers.equal(sentence_out, pad)), 'float32')
            sentence_in = layers.unsqueeze(sentence_in, axes=[2])

            x = self._embedding(sentence_in)
            loss = 0
            for t in range(_configuration['sentence_length'] - 1):
                context, alpha = self._attention_layer(feature, feature_proj, h)

                if _configuration['enhancement']['gating_scalar']:
                    context, beta = self._selector(context, h)
                
                feed = layers.concat([x[:, t, :], context], axis=1)
                c, h = self._lstm_unit(feed, c, h)
                logits = self._decode_lstm(x[:, t, :], h, context, _configuration['enhancement']['dropout'])
                batch_loss = self._calc_loss(logits, sentence_out[:, t]) * padding_mask[:, t]
                loss += layers.mean(batch_loss)
            return loss
        else: # infer
            result = []
            word = layers.fill_constant_batch_size_like(image, [1, ], 'int64', _configuration['start_idx'])
            for t in range(_configuration['infer_max_length']):
                word = layers.unsqueeze(word, axes=[1])
                x = self._embedding(word)
                context, alpha = self._attention_layer(feature, feature_proj, h)

                if _configuration['enhancement']['gating_scalar']:
                    context, beta = self._selector(context, h)
                
                feed = layers.concat([x, context], axis=1)
                c, h = self._lstm_unit(feed, c, h)
                logits = self._decode_lstm(x, h, context, _configuration['enhancement']['dropout'])
                word = layers.argmax(logits, axis=-1)
                result.append(word)

            result = layers.stack(result)
            return layers.transpose(result, [1, 0])
        
    def _calc_loss(self, logits, real):
        label = fluid.layers.unsqueeze(real, axes=[1])
        # 注意这里的stop_gradient,感觉像是一个bug
        label.stop_gradient = True
        loss = fluid.layers.softmax_with_cross_entropy(logits, label, axis=-1)
        return loss

    def _decode_lstm(self, x, h, context, dropout=False):
        if dropout:
            h = layers.dropout(h, 0.5)
        logits = self._decode_hidden_proj(h)
        ctx2out = layers.matmul(context, self.decode_ctx2out)
        logits += ctx2out
        logits += x
        logits = self._decode_output(logits)
        return logits

    def _lstm_unit(self, x, c_prev, h_prev, forget_bias=0.0):
        '''
        见 `fluid.nn.layers.lstm_unit`
        '''
        concat = layers.concat([x, h_prev], axis=1)
        fc_out = self._lstm_fc(concat)
        c = self._helper.create_variable_for_type_inference('float32')
        h = self._helper.create_variable_for_type_inference('float32')

        self._helper.append_op(
            type='lstm_unit',
            inputs={"X": fc_out,
                    "C_prev": c_prev},
            outputs={"C": c,
                    "H": h},
            attrs={"forget_bias": forget_bias}
        )
        
        return c, h

    def _get_initial_lstm(self, feature):
        feature_mean = layers.reduce_mean(feature, dim=1)
        h = self._initial_lstm_h(feature_mean)
        c = self._initial_lstm_c(feature_mean)
        return c, h

    def _attention_layer(self, feature, feature_proj, h):
        hidden_proj = layers.unsqueeze(self._att_hidden_proj(h), [1])
        hidden_att = layers.elementwise_add(feature_proj, hidden_proj, 0)
        out_att = layers.matmul(hidden_att, self.att_w)
        out_att = layers.squeeze(out_att, [2])
        alpha = layers.softmax(out_att, use_cudnn=_configuration['use_cudnn'])

        context = layers.elementwise_mul(feature, alpha, axis=0)
        context = layers.reduce_sum(context, dim=1)

        return context, alpha

    def _selector(self, context, hidden):
        hidden_proj = self._selector_hidden_proj(hidden)
        beta = layers.sigmoid(hidden_proj)
        context = layers.elementwise_mul(context, beta, axis=0)
        return context, beta

    def _feature_proj(self, feature):
        return layers.matmul(feature, self.feature_proj_w)

    def createParam(self, shape, dtype='float32', name=None, 
        is_bias=False, initializer=None):
        '''
        创建一个参数
        '''
        if name is None:
            if is_bias:
                self._bias_count += 1
                name = '_b%d' % self._bias_count
            else:
                self._weight_count += 1
                name = '_w%d' % self._weight_count
        attr = fluid.ParamAttr(name=name)
        # 翻查源码后发现，这里没有必要调用add_parameter
        # 原因请看dygraph.layers.__setattr__
        # 我就说怎么FC里的weight调用了而bias没有调用
        # return self.add_parameter(name, self.create_parameter(attr, shape, dtype, is_bias, initializer))
        return self.create_parameter(attr, shape, dtype, is_bias, initializer)


if __name__ == "__main__":
    from dy_mobilenetV2 import MobileNetV2 as m2
    global MobileNetV2
    MobileNetV2 = m2

    def static_test():
        x = np.ones([1, 3, 224, 224], dtype='float32')
        y = np.zeros([1, 39], dtype='int64')

        exe = fluid.Executor(fluid.CPUPlace())
        feed = fluid.layers.data('img', [None, 3, 224, 224], dtype='float32')
        sentence = fluid.layers.data('sentence', [None, 39], dtype='int64')

        layer = MobileNetV2('aaa')
        x = layer(feed)

        exe.run(fluid.default_startup_program())
        out = exe.run(fluid.default_main_program(), feed={'img': x, 'sentence': y}, fetch_list=[x])
        # fluid.io.save_persistables(exe, r"./save/")

    def dynamic_test():

        with dygraph.guard():
            layer = ImageCaptioning('test')
            x = dygraph.to_variable(np.ones([16, 3, 224, 224], dtype='float32'))
            y = dygraph.to_variable(np.zeros([16, 39], dtype='int64'))
            
            out = layer(x, y)
            out.backward()
            layer.feature_proj_w.gradient()
            print('finish')
            print(out.numpy())

    static_test()
