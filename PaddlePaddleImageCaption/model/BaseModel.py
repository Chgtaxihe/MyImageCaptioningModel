import paddle.fluid as fluid
from model import MobileNetV2
from paddle.fluid.param_attr import ParamAttr

# ref:show-attend-and-tell

_configuration = {
    'vocab_size': 18211, # 字典大小
    'embedding_size': 512,
    'sentence_length': 44, # 训练用句子的最长长度(包括<start>和<end>)
    'padding_idx': 0,
    'start_idx': 2, # `<start>`` 的id
    'encoder_dim': 7 * 7,
    'encoder_channel': 1280, 
    'hidden_dim': 1024,
    'enhancement':{
        'doubly_stochastic_attention': False, # 见论文(4.2.1) 还未实现！！
        'gating_scalar': True, # 见论文(4.2.1)
        'dropout': False # 开了dropout貌似容易Nan
    },
    'use_cudnn': False,
    'infer_max_length': 44,
    'lstm_weight_clip': None,
    'lstm_bias_clip': None
}

def attr(name):
    return ParamAttr(name=name)

class ImageCaptionModle():

    def __init__(self, use_cudnn=False, cnn_trainable=False):
        self.M = _configuration['embedding_size']
        self.V = _configuration['vocab_size']
        self.L = _configuration['encoder_dim']
        self.D = _configuration['encoder_channel']
        self.H = _configuration['hidden_dim']
        self.n_time_step = _configuration['sentence_length']
        self._null = _configuration['padding_idx']
        self._begin = _configuration['start_idx']
        self.use_cudnn = use_cudnn
        self.cnn_trainable = cnn_trainable

        self.ctx_proj = self._variable([self.D, self.M], 'decode_lstm_ctx_proj_w')
        self.att_out_w = self._variable([self.D, 1], name='attention_out_w')
        self.feature_proj_w = self._variable([self.D, self.D], name='feature_proj_w')

    def build_sampler(self, image, max_len=None):
        if max_len is None:
            max_len = _configuration['infer_max_length']
        
        # encoder = MobileNetV2.MobileNetV2(trainable=self.cnn_trainable)
        # feature = encoder.net(image)
        feature = image
        # feature = fluid.layers.batch_norm(feature)
        feature = fluid.layers.reshape(feature, [-1, self.D, self.L])
        feature = fluid.layers.transpose(feature, [0, 2, 1]) # 把Channel放在最后

        # lstm states
        c, h = self.get_inital_lstm(feature)
        feature_proj = self.feature_project(feature)

        word_list = []
        batch_size = fluid.layers.shape(image)[0]
        word = fluid.layers.fill_constant([batch_size], dtype='int64', value=self._begin)
        for idx in range(max_len):
            word_vector = self.embedding(word)

            context, alpha = self.attention_layer(feature, feature_proj, h)

            if _configuration['enhancement']['gating_scalar']:
                context, beta = self.selector(context, h)

            x = fluid.layers.concat([word_vector, context], axis=1)
            h, c = self.lstm_unit(x, h, c)
            logits = self.decode_lstm(word_vector, h, context)
            word = fluid.layers.argmax(logits, axis=-1)
            word_list.append(word)

        result = fluid.layers.stack(word_list)

        return fluid.layers.transpose(result, [1, 0])

    def build_network(self, image, sentence):
        sentence_in = sentence[:, :-1]
        sentence_out = sentence[:, 1:]

        sentence_shape = fluid.layers.shape(sentence_out)

        mask = fluid.layers.equal(sentence_out, 
            fluid.layers.fill_constant(shape=sentence_shape, dtype='int64', value=self._null))
        mask = fluid.layers.cast(fluid.layers.logical_not(mask), 'float32')

        # image encoding
        # encoder = MobileNetV2.MobileNetV2(trainable=self.cnn_trainable)
        # feature = encoder.net(image)
        feature = image
        # feature = fluid.layers.batch_norm(feature)
        feature = fluid.layers.reshape(feature, [-1, self.D, self.L])
        feature = fluid.layers.transpose(feature, [0, 2, 1]) # 把Channel放在最后

        # word embedding
        word_vector = self.embedding(sentence_in)

        # lstm states
        c, h = self.get_inital_lstm(feature)
        feature_proj = self.feature_project(feature)

        loss = 0
        for t in range(self.n_time_step - 1):
            context, alpha = self.attention_layer(feature, feature_proj, h)

            if _configuration['enhancement']['gating_scalar']:
                context, beta = self.selector(context, h)

            x = fluid.layers.concat([word_vector[:, t, :], context], axis=1)
            h, c = self.lstm_unit(x, h, c)
            logits = self.decode_lstm(word_vector[:, t, :], h, context)
            batch_loss = self.calc_loss(logits, sentence_out[:, t]) * mask[:, t]
            # paddle规定最终的loss必须是标量
            loss += fluid.layers.mean(batch_loss)

        return loss

    def selector(self, context, h, name='selector'):
        hidden_proj = self.fc(h, 1, name=name+'_hidden_proj')
        beta = fluid.layers.sigmoid(hidden_proj)
        context = fluid.layers.elementwise_mul(context, beta, axis=0)
        return context, beta

    def embedding(self, x):
        return fluid.embedding(x, 
            [self.V, self.M],
            is_sparse=True,
            param_attr=ParamAttr(name='word_embedding', 
                initializer=fluid.initializer.Uniform()))

    def calc_loss(self, logits, ground):
        label = fluid.layers.unsqueeze(ground, axes=[1])
        # 注意这里的stop_gradient,感觉像是一个bug
        label.stop_gradient = True
        loss = fluid.layers.softmax_with_cross_entropy(logits, label, axis=-1)
        return loss
    
    def lstm_unit(self, x, h, c, name='lstm_loop'):
        weight_attr = ParamAttr(name=name+'_weight')
        bias_attr = ParamAttr(name=name+'_bias')
        if _configuration['lstm_weight_clip'] is not None:
            weight_attr.gradient_clip = fluid.clip.GradientClipByValue(_configuration['lstm_weight_clip'])
        if _configuration['lstm_bias_clip'] is not None:
            bias_attr.gradient_clip = fluid.clip.GradientClipByValue(_configuration['lstm_bias_clip'])

        return fluid.layers.lstm_unit(x, h, c,
            param_attr=weight_attr, bias_attr=bias_attr)

    def get_inital_lstm(self, image_feature, name='initial_lstm'):
        feature_mean = fluid.layers.reduce_mean(image_feature, dim=1)
        h = self.fc(feature_mean, self.H, name=name+'_hidden', act='tanh')
        c = self.fc(feature_mean, self.H, name=name+'_cell', act='tanh')
        return c, h

    def feature_project(self, image_feature):
        image_feature = fluid.layers.matmul(image_feature, self.feature_proj_w)
        return image_feature

    # paddle 是根据 parameter的name 来进行判断是否是同一个参数，不要指定reuse参数
    def attention_layer(self, feature, feature_proj, h, name='attention'):
        hidden_proj = self.fc(h, self.D, 1, name=name+'_hidden_proj')

        h_att = fluid.layers.relu(feature_proj + fluid.layers.unsqueeze(hidden_proj, [1]))
        out_att = fluid.layers.matmul(h_att, self.att_out_w)
        out_att = fluid.layers.squeeze(out_att, axes=[2])
        alpha = fluid.layers.softmax(out_att, use_cudnn=self.use_cudnn)

        # Bahdanau attention
        context = fluid.layers.elementwise_mul(feature, alpha, axis=0)
        context = fluid.layers.reduce_sum(context, dim=1)
        return context, alpha

    def decode_lstm(self, x, h, context, name='decode_lstm'):
        if _configuration['enhancement']['dropout']:
            h = fluid.layers.dropout(h, 0.5)
        h_logits = self.fc(h, self.M, name=name+'_hidden_proj')
        
        ctx = fluid.layers.matmul(context, self.ctx_proj)
        h_logits += ctx 
        h_logits += x

        h_logits = fluid.layers.tanh(h_logits)
        if _configuration['enhancement']['dropout']:
            h_logits = fluid.layers.dropout(h_logits, 0.5)

        out_logits = self.fc(h_logits, self.V, name=name+'_out')
        return out_logits
    
    def _variable(self, shape, name, dtype='float32', is_bias=False):
        return fluid.layers.create_parameter(shape, dtype, attr=attr(name), is_bias=is_bias)

    def fc(self, x, size, num_flatten_dims=1, name='', act=None):
        return fluid.layers.fc(x, size, num_flatten_dims=num_flatten_dims,
            param_attr=attr(name+'_w'), bias_attr=attr(name+'_b'), act=act)
    

if __name__ == "__main__":
    import numpy as np
    import paddle.fluid.layers as layers
    fake_data = np.ones([2, 16], dtype='float32')

    w = layers.create_parameter([16, 7], dtype='float32')
    x = layers.data('val', [None, 16])
    x = layers.matmul(x, w)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    out = exe.run(fluid.default_main_program(), feed={'val': fake_data}, fetch_list=[x])
    print(out)
