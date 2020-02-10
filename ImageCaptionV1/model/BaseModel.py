from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.param_attr import ParamAttr

# ref:show-attend-and-tell

_configuration = {
    'vocab_size': 18211,  # 字典大小
    'embedding_size': 512,
    'sentence_length': 44,  # 训练用句子的最长长度(包括<start>和<end>)
    'padding_idx': 0,
    'start_idx': 2,  # `<start>`` 的id
    'encoder_dim': 7 * 7,
    'encoder_channel': 1280,
    'hidden_dim': 1024,
    'enhancement': {
        'doubly_stochastic_attention': False,  # 见论文(4.2.1) 还未实现！！
        'gating_scalar': True,  # 见论文(4.2.1)
    },
    'use_cudnn': False,
    'infer_max_length': 44,
}
_conf = _configuration


def variable(shape, name, dtype='float32', is_bias=False):
    return layers.create_parameter(shape, dtype, attr=ParamAttr(name), is_bias=is_bias)


def fc(x, size, num_flatten_dims=1, name='', act=None):
    return layers.fc(x, size, num_flatten_dims=num_flatten_dims,
                     param_attr=ParamAttr(name + '_w'), bias_attr=ParamAttr(name + '_b'), act=act)


class ImageCaptionModel:

    def __init__(self, use_cudnn=True):
        self.M = _configuration['embedding_size']
        self.V = _configuration['vocab_size']
        self.L = _configuration['encoder_dim']
        self.D = _configuration['encoder_channel']
        self.H = _configuration['hidden_dim']
        self.n_time_step = _configuration['sentence_length']
        self._null = _configuration['padding_idx']
        self._begin = _configuration['start_idx']
        self.use_cudnn = use_cudnn

        self.ctx_proj = variable([self.D, self.M], 'decode_lstm_ctx_proj_w')
        self.att_out_w = variable([self.D, 1], name='attention_out_w')
        self.feature_proj_w = variable([self.D, self.D], name='feature_proj_w')
        # self.lstm_h_w = variable([_conf['hidden_dim'], 4 * _conf['hidden_dim']], name='lstm_h_w')
        # self.lstm_x_w = variable([_conf['encoder_channel'] + _conf['embedding_size'], 4 * _conf['hidden_dim']],
        #                          name='lstm_x_w')
        # self.lstm_bias = variable([4 * _conf['hidden_dim']], name='lstm_bias')

    def eval_network(self, image, max_len=None):
        if max_len is None:
            max_len = _configuration['infer_max_length']

        feature = image
        feature = layers.reshape(feature, [-1, self.D, self.L])
        feature = layers.transpose(feature, [0, 2, 1])  # 把Channel放在最后

        # lstm states
        c, h = self.get_initial_lstm(feature)
        feature_proj = self.feature_project(feature)

        word_list = []
        batch_size = layers.shape(image)[0]
        word = layers.fill_constant([batch_size], dtype='int64', value=self._begin)
        for idx in range(max_len):
            word_vector = self.embedding(word)

            context, alpha = self.attention_layer(feature, feature_proj, h)

            if _configuration['enhancement']['gating_scalar']:
                context, beta = self.selector(context, h)

            x = layers.concat([word_vector, context], axis=1)
            h, c = self.lstm_unit(x, h, c)
            logits = self.decode_lstm(word_vector, h, context)
            word = layers.argmax(logits, axis=-1)
            word_list.append(word)

        result = layers.stack(word_list)

        return layers.transpose(result, [1, 0])

    def training_network(self, image, sentence):
        sentence_in = sentence[:, :-1]
        sentence_out = sentence[:, 1:]

        padding_filled = layers.fill_constant_batch_size_like(sentence_out, shape=[-1, _conf['sentence_length'] - 1],
                                                              dtype='int64', value=_conf['padding_idx'])
        mask = layers.equal(sentence_out, padding_filled)
        mask = layers.cast(layers.logical_not(mask), 'float32')
        mask.stop_gradient = True

        feature = image
        feature = layers.reshape(feature, [-1, self.D, self.L])
        feature = layers.transpose(feature, [0, 2, 1])  # 把Channel放在最后

        # word embedding
        word_vector = self.embedding(sentence_in)

        # lstm states
        c, h = self.get_initial_lstm(feature)
        feature_proj = self.feature_project(feature)

        loss = 0
        for t in range(self.n_time_step - 1):
            context, alpha = self.attention_layer(feature, feature_proj, h)
            # context = layers.reduce_mean(feature, dim=1)
            if _configuration['enhancement']['gating_scalar']:
                context, beta = self.selector(context, h)
            x = layers.concat([word_vector[:, t, :], context], axis=1)
            h, c = self.lstm_unit(x, h, c)
            logits = self.decode_lstm(word_vector[:, t, :], h, context)
            batch_loss = self.calc_loss(logits, sentence_out[:, t]) * mask[:, t]
            # paddle规定最终的loss必须是标量
            loss += layers.mean(batch_loss)

        return loss

    def selector(self, context, h, name='selector'):
        hidden_proj = fc(h, 1, name=name + '_hidden_proj')
        beta = layers.sigmoid(hidden_proj)
        context = layers.elementwise_mul(context, beta, axis=0)
        return context, beta

    @staticmethod
    def embedding(x):
        return fluid.embedding(x,
                               [_conf['vocab_size'], _conf['embedding_size']],
                               is_sparse=True,
                               padding_idx=_conf['padding_idx'],
                               param_attr=ParamAttr(name='word_embedding',
                                                    initializer=fluid.initializer.Uniform()))

    @staticmethod
    def calc_loss(prediction, ground):
        label = layers.unsqueeze(ground, axes=[1])
        label.stop_gradient = True
        loss = layers.softmax_with_cross_entropy(prediction, label, axis=-1)
        return loss

    def lstm_unit(self, x, h, c, name='lstm_loop'):
        return layers.lstm_unit(x, h, c,
                                param_attr=ParamAttr(name=name + '_w'),
                                bias_attr=ParamAttr(name=name + '_b'))
        # x = layers.matmul(x, self.lstm_x_w)
        # h = layers.matmul(h, self.lstm_h_w)
        # s = x + h + self.lstm_bias
        # forget_gate, input_gate, output_gate, cell = layers.split(s, 4)
        # cell = layers.elementwise_mul(forget_gate, c) + \
        #       layers.elementwise_mul(input_gate, layers.tanh(cell))
        # hidden = layers.elementwise_mul(output_gate, cell)
        # return hidden, cell

    @staticmethod
    def get_initial_lstm(image_feature, name='initial_lstm'):
        feature_mean = layers.reduce_mean(image_feature, dim=1)
        h = fc(feature_mean, _conf['hidden_dim'], name=name + '_hidden', act='tanh')
        c = fc(feature_mean, _conf['hidden_dim'], name=name + '_cell', act='tanh')
        return c, h

    def feature_project(self, image_feature):
        image_feature = layers.matmul(image_feature, self.feature_proj_w)
        return image_feature

    # paddle 是根据 parameter的name 来进行判断是否是同一个参数，不要指定reuse参数
    # def attention_layer(self, feature, feature_proj, h, name='attention'):
    #     hidden_proj = fc(h, self.D, 1, name=name + '_hidden_proj')
    #
    #     # 下面这行代码使得一个Epoch的耗时增加了500多秒
    #     h_att = layers.relu(feature_proj + layers.unsqueeze(hidden_proj, [1]))
    #
    #     out_att = layers.matmul(h_att, self.att_out_w)
    #     out_att = layers.squeeze(out_att, axes=[2])
    #     alpha = layers.softmax(out_att, use_cudnn=self.use_cudnn)
    #
    #     # Bahdanau attention
    #     context = layers.elementwise_mul(feature, alpha, axis=0)
    #     context = layers.reduce_sum(context, dim=1)
    #     return context, alpha

    def attention_layer(self, feature, feature_proj, h, name='attention'):
        feature_proj = fc(layers.reduce_mean(feature_proj, dim=1), _conf['hidden_dim'], 1,
                          name=name+'fea_proj', act='tanh')
        hidden_proj = fc(h, _conf['hidden_dim'], 1, name=name+'_fc1', act='tanh')
        hidden_proj2 = fc(hidden_proj + feature_proj, _conf['encoder_dim'], 1, name=name+'_fc2')
        alpha = layers.softmax(hidden_proj2, use_cudnn=self.use_cudnn)

        # Bahdanau attention
        context = layers.elementwise_mul(feature, alpha, axis=0)
        context = layers.reduce_sum(context, dim=1)
        return context, alpha

    def decode_lstm(self, x, h, context, name='decode_lstm'):
        h_logits = fc(h, _conf['embedding_size'], name=name + '_hidden_proj')

        ctx = layers.matmul(context, self.ctx_proj)
        h_logits += ctx
        h_logits += x

        h_logits = layers.tanh(h_logits)
        out_logits = fc(h_logits, _conf['vocab_size'], name=name + '_out')
        return out_logits
