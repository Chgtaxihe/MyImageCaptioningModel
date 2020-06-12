import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid import ParamAttr

import config
from model.MobileNetV2 import MobileNetV2

encoder_config = config.md['encoder']
decoder_config = config.md['decoder']

orthogonal_init_vars = []


class DecoderCell(layers.RNNCell):

    def call(self, inputs, states, **kwargs):
        hidden_prev, cell_prev = layers.split(states, 2)
        hidden, cell = layers.lstm_unit(inputs, hidden_prev, cell_prev,
                                        param_attr=ParamAttr('lstm_w',
                                                             initializer=fluid.initializer.TruncatedNormal(scale=0.01),
                                                             gradient_clip=fluid.clip.GradientClipByGlobalNorm(0.05)),
                                        bias_attr=ParamAttr('lstm_b'))
        next_state = layers.concat([hidden, cell], axis=-1)
        return hidden, next_state

    @staticmethod
    def get_init_states(batch_ref):
        hidden = layers.fill_constant_batch_size_like(batch_ref,
                                                      shape=[0, decoder_config['hidden_dim']],
                                                      dtype='float32',
                                                      value=0)
        memory = layers.fill_constant_batch_size_like(batch_ref,
                                                      shape=[0, decoder_config['hidden_dim']],
                                                      dtype='float32',
                                                      value=0)
        return layers.concat([hidden, memory], axis=-1)


class ImageCaptionModel:

    def __init__(self, use_raw_image=False):
        self.use_raw_image = use_raw_image

        if use_raw_image:
            self.encoder = MobileNetV2(trainable=encoder_config['encoder_trainable'], use_pooling=True)

    def build_input(self, mode='train'):
        if mode not in ['train', 'eval']:
            raise ValueError('不支持{}'.format(mode))
        if config.train['use_raw_image']:
            img = fluid.data('image', [-1, 3, 224, 224])
        else:
            img = fluid.data('image', [-1, 1280])
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

    def final_fc(self, inputs):
        embedding_w = layers.create_parameter(shape=[decoder_config['vocab_size'], decoder_config['embedding_size']],
                                              dtype='float32',
                                              attr=ParamAttr(name='word_embedding',
                                                             initializer=fluid.initializer.Uniform()))
        bias = layers.create_parameter([decoder_config['vocab_size']],
                                       dtype='float32',
                                       is_bias=True,
                                       name='out_fc_bias')
        return layers.elementwise_add(layers.matmul(inputs, embedding_w, transpose_y=True), bias)

    def training_network(self, img, caption):
        image_embedding = self._img2feature(img)

        # build caption and mask
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

        # model
        rnn_cell = DecoderCell()
        _, initial_state = rnn_cell.call(image_embedding, rnn_cell.get_init_states(image_embedding))

        output, _ = layers.rnn(cell=rnn_cell,
                               inputs=source,
                               initial_states=initial_state)
        # output [batch_size, seq_len, hidden_dim]
        final_output = self.final_fc(output)
        loss = layers.squeeze(ImageCaptionModel.loss(target, final_output), axes=[2])
        loss = layers.elementwise_mul(loss, mask)

        return layers.reduce_sum(loss) / scale_factor

    def eval_network(self, img):
        image_embedding = self._img2feature(img)
        rnn_cell = DecoderCell()
        embedding_fn = lambda a: fluid.embedding(a, (decoder_config['vocab_size'], decoder_config['embedding_size']),
                                                 padding_idx=config.data['padding_idx'],
                                                 param_attr=ParamAttr(name='word_embedding',
                                                                      initializer=fluid.initializer.Uniform()))
        mode = decoder_config['infer_mode']
        if mode not in ['normal']:
            raise ValueError('Infer mode {} is not supported!'.format(mode))

        if mode == 'normal':
            mx_step = layers.fill_constant([1], 'int64', value=decoder_config['infer_max_length'])
            step_idx = layers.fill_constant([1], 'int64', value=0)
            cond = layers.less_equal(step_idx, mx_step)
            while_op = layers.While(cond)

            output_array = layers.create_array('int64')
            word = layers.fill_constant_batch_size_like(image_embedding, [0], 'int64', value=config.data['start_idx'])
            state = rnn_cell.get_init_states(image_embedding)
            _, state = rnn_cell.call(image_embedding, state)
            with while_op.block():
                embed = embedding_fn(word)
                output, next_state = rnn_cell.call(embed, state)  # [batch_size, vocab_size], [batch_size, hidden_dim]
                output = self.final_fc(output)
                w = layers.argmax(output, axis=-1)  # [batch_size]
                layers.array_write(w, step_idx, array=output_array)

                layers.increment(step_idx)
                layers.assign(w, word)
                layers.assign(next_state, state)

                layers.less_equal(step_idx, mx_step, cond=cond)

            rnn_output, _ = layers.tensor_array_to_tensor(output_array, axis=0, use_stack=True)
            result = layers.transpose(rnn_output, [1, 0])

        return result

    def _img2feature(self, img):
        if self.use_raw_image:
            img = self.encoder.net(img)
            img = layers.squeeze(img, axes=[2, 3])
        img_embedding = layers.fc(img, decoder_config['embedding_size'], name='img_embedding', bias_attr=False)
        return img_embedding

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
