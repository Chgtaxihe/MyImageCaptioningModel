import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid import ParamAttr

import config
from model.MobileNetV2 import MobileNetV2

encoder_config = config.md['encoder']
decoder_config = config.md['decoder']

orthogonal_init_vars = ['lstm_w']


class DecoderCell(layers.RNNCell):

    def call(self, inputs, states, **kwargs):
        hidden_prev, cell_prev = layers.split(states, 2)

        feedin = inputs
        hidden, cell = layers.lstm_unit(feedin, hidden_prev, cell_prev,
                                        param_attr=ParamAttr('lstm_w'),
                                        bias_attr=ParamAttr('lstm_b'))
        next_state = layers.concat([hidden, cell], axis=-1)
        output = layers.fc(hidden, decoder_config['vocab_size'], num_flatten_dims=1)
        return output, next_state

    @staticmethod
    def get_init_states(feature):
        # feature [batch_size, 7, 7, channel]
        # 注：这里的dim写错了！！应为dim=[2, 3], 相当于global average pooling
        feature = layers.reduce_mean(feature, dim=1)
        hidden = layers.fc(feature, decoder_config['hidden_dim'], param_attr=ParamAttr('init_lstm_h_w'),
                           bias_attr=ParamAttr('init_lstm_h_b'), num_flatten_dims=1)
        memory = layers.fc(feature, decoder_config['hidden_dim'], param_attr=ParamAttr('init_lstm_c_w'),
                           bias_attr=ParamAttr('init_lstm_c_b'), num_flatten_dims=1)
        return layers.concat([hidden, memory], axis=-1)


class ImageCaptionModel:

    def __init__(self, use_raw_image=False):
        self.use_raw_image = use_raw_image

        if use_raw_image:
            self.encoder = MobileNetV2(trainable=encoder_config['encoder_trainable'])

    def build_input(self, mode='train'):
        if mode not in ['train', 'eval']:
            raise ValueError('不支持{}'.format(mode))

        if config.train['use_raw_image']:
            img = fluid.data('image', [-1, 3, 224, 224])
        else:
            img = fluid.data('image', [-1, encoder_config['encoder_channel'], 7, 7])
        if mode == 'train':
            caption = fluid.data('caption', [-1, decoder_config['sentence_length']], dtype='int64')
            return {'img': img, 'caption': caption}, [img, caption]
        return {'img': img}, [img]

    def build_network(self, mode='train', **kwargs):
        if mode not in ['train', 'eval']:
            raise ValueError('不支持{}'.format(mode))
        if mode == 'train':
            return self.training_network(**kwargs)
        elif mode == 'eval':
            return self.eval_network(**kwargs)

    def training_network(self, img, caption):
        feature = self._img2feature(img)
        target = caption[:, 1:]

        source = caption[:, :-1]
        source = fluid.embedding(source, (decoder_config['vocab_size'], decoder_config['embedding_size']),
                                 padding_idx=config.data['padding_idx'],
                                 param_attr=ParamAttr(name='word_embedding',
                                                      initializer=fluid.initializer.Uniform()))

        padding_filled = layers.fill_constant_batch_size_like(target, shape=[-1, decoder_config['sentence_length'] - 1],
                                                              dtype='int64', value=config.dc['padding_idx'])
        mask = layers.equal(target, padding_filled)
        mask = layers.cast(layers.logical_not(mask), 'float32')
        scale_factor = layers.reduce_sum(mask)
        mask.stop_gradient = True
        scale_factor.stop_gradient = True

        rnn_cell = DecoderCell()
        final_output, _ = layers.rnn(cell=rnn_cell,
                                     inputs=source,
                                     initial_states=rnn_cell.get_init_states(feature),
                                     encoder_output=feature)
        loss = layers.squeeze(ImageCaptionModel.loss(target, final_output), axes=[2])
        loss = layers.elementwise_mul(loss, mask)

        return layers.reduce_sum(loss) / scale_factor

    def eval_network(self, img):
        feature = self._img2feature(img)
        rnn_cell = DecoderCell()
        embedding_fn = lambda a: fluid.embedding(a, (decoder_config['vocab_size'], decoder_config['embedding_size']),
                                                 padding_idx=config.data['padding_idx'],
                                                 param_attr=ParamAttr(name='word_embedding',
                                                                      initializer=fluid.initializer.Uniform()))
        mode = decoder_config['infer_mode']
        if mode not in ['normal', 'beam_search']:
            raise ValueError('Infer mode {} is not supported!'.format(mode))

        if mode == 'normal':
            mx_step = layers.fill_constant([1], 'int64', value=decoder_config['infer_max_length'])
            step_idx = layers.fill_constant([1], 'int64', value=0)
            cond = layers.less_equal(step_idx, mx_step)
            while_op = layers.While(cond)

            output_array = layers.create_array('int64')
            word = layers.fill_constant_batch_size_like(feature, [0], 'int64', value=config.data['start_idx'])
            state = rnn_cell.get_init_states(feature)

            with while_op.block():
                embed = embedding_fn(word)
                output, next_state = rnn_cell.call(embed, state)  # [batch_size, vocab_size], [batch_size, hidden_dim]
                w = layers.argmax(output, axis=-1)  # [batch_size]
                layers.array_write(w, step_idx, array=output_array)

                layers.increment(step_idx)
                layers.assign(w, word)
                layers.assign(next_state, state)

                layers.less_equal(step_idx, mx_step, cond=cond)

            rnn_output, _ = layers.tensor_array_to_tensor(output_array, axis=0, use_stack=True)
            result = layers.transpose(rnn_output, [1, 0])

        elif mode == 'beam_search':
            beam_search = layers.BeamSearchDecoder(rnn_cell,
                                                   start_token=config.data['start_idx'],
                                                   end_token=config.data['stop_idx'],
                                                   beam_size=decoder_config['beam_size'],
                                                   embedding_fn=embedding_fn)
            outputs, final_state = layers.dynamic_decode(beam_search, rnn_cell.get_init_states(feature),
                                                         max_step_num=decoder_config['infer_max_length'])
            result = outputs[:, :, 0]

        return result

    def _img2feature(self, img):
        if self.use_raw_image:
            img = self.encoder.net(img)
        img = layers.reshape(img, shape=[0, encoder_config['encoder_channel'], encoder_config['encoder_dim']],
                             inplace=True)
        img = layers.transpose(img, [0, 2, 1])  # channel last
        return img

    @staticmethod
    def first_init(places):
        # 正交初始化 https://blog.csdn.net/weixin_40485472/article/details/81203314
        def orthogonal(shape):
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            return q.reshape(shape).astype('float32')

        for name in orthogonal_init_vars:
            var = fluid.global_scope().find_var(name).get_tensor()
            var_shape = np.array(var).shape
            var.set(orthogonal(var_shape), places)

    @staticmethod
    def loss(ground_true, prediction):
        # ground_true: [batch_size, seq_len]
        # prediction: [batch_size, seq_len, vocab_size]
        ground_true = layers.unsqueeze(ground_true, axes=2)
        ground_true.stop_gradient = True
        los = layers.softmax_with_cross_entropy(prediction, ground_true, axis=-1)
        return los

# baseline_5:
# 存在问题: generic caption, Nan
# seed: 107131, mx_len: 44
# Best BLEU
# Meteor 分数: 0.2424676
# BLEU [0.502980, 0.216131, 0.212258, 0.186382] Mean: 0.2794379
# Best Meteor
# Meteor 分数: 0.2485872
# BLEU [0.456206, 0.197677, 0.215573, 0.196412] Mean: 0.2664670
# baseline_1:
# 存在问题: generic caption
# seed: 107131, mx_len: 44
# Meteor 分数: 0.2340070
# Best BLEU
# BLEU [0.416870, 0.196700, 0.216145, 0.195523] Mean: 0.2563097

# baseline_5:
# 存在问题: generic caption
# seed: 666666, mx_len: 34
# Meteor 分数: 0.2423692
# BLEU [0.499315, 0.215799, 0.212425, 0.186427] Mean 0.2784915
