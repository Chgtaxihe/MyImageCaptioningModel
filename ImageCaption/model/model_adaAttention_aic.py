import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid import ParamAttr

import config
from model.MobileNetV2 import MobileNetV2

encoder_config = config.md['encoder']
decoder_config = config.md['decoder']

orthogonal_init_vars = []


def weight_tying_fc(x):
    embedding_w = layers.create_parameter(shape=[decoder_config['vocab_size'], decoder_config['embedding_size']],
                                          dtype='float32',
                                          attr=ParamAttr(name='word_embedding',
                                                         initializer=fluid.initializer.Uniform()))
    bias = layers.create_parameter([decoder_config['vocab_size']],
                                   dtype='float32',
                                   is_bias=True,
                                   name='out_fc_bias')
    proj_x = layers.fc(x, decoder_config['embedding_size'])
    return layers.elementwise_add(layers.matmul(proj_x, embedding_w, transpose_y=True), bias)


def embedding_function(x):
    return fluid.embedding(x, (decoder_config['vocab_size'], decoder_config['embedding_size']),
                           padding_idx=config.data['padding_idx'],
                           param_attr=ParamAttr(name='word_embedding',
                                                initializer=fluid.initializer.Uniform()))


def create_zero_state(batch_ref):
    return layers.fill_constant_batch_size_like(batch_ref,
                                                shape=[-1, decoder_config['hidden_dim']], dtype='float32', value=0)


class Decoder:

    def __init__(self, hidden_size, mode='train', rnn_layer=1):
        if mode not in ['train', 'eval']:
            raise ValueError('不支持{}'.format(mode))
        self.hid_size = hidden_size
        self.rnn_layer = rnn_layer
        self.mode = mode
        self.start_tag = config.data['start_idx']

    def call(self, global_img_feat, p_img_feat, embedding_fn, words=None):
        # 图片特征
        img_feat = layers.fc(p_img_feat, self.hid_size, num_flatten_dims=2, act='tanh')  # [batch, k, hid]
        img_feat_emb = layers.fc(p_img_feat, self.hid_size, num_flatten_dims=2)

        if self.mode == 'eval':
            word = layers.fill_constant_batch_size_like(global_img_feat, [-1],
                                                        dtype='int64',
                                                        value=config.data['start_idx'])
        else:
            words = layers.transpose(words, [1, 0])  # [seq, batch]
            words.stop_gradient = True
        # lstm 初始化
        hid, cell = create_zero_state(global_img_feat), create_zero_state(global_img_feat)

        # While loop 参数初始化
        mx = decoder_config['sentence_length'] - 1 if self.mode == 'train' else decoder_config['infer_max_length']
        if self.mode == 'eval':
            mx = decoder_config['infer_max_length']
            while_op_output = layers.create_array('int64')
        else:
            while_op_output = layers.create_array('float32')
        max_step = layers.fill_constant(shape=[1], dtype='int64', value=mx)
        step = layers.fill_constant(shape=[1], dtype='int64', value=0)
        cond = layers.less_than(step, max_step)
        while_op = layers.While(cond)

        with while_op.block():
            if self.mode == 'train':
                st = layers.cast(step, 'int32')
                word = layers.slice(words, axes=[0], starts=st, ends=st + 1)
                word = layers.squeeze(word, [0])
                word.stop_gradient = True

            word_emb = embedding_fn(word)
            # 这里可能用+效果更好？
            xt = layers.concat([word_emb, global_img_feat], axis=-1)  # [batch, feat]
            h, c = layers.lstm_unit(xt, hid, cell, param_attr=fluid.ParamAttr('lstm_w'),
                                    bias_attr=fluid.ParamAttr('lstm_b'))
            p_word_emb = layers.fc(xt, size=self.hid_size)
            p_hidden = layers.fc(hid, size=self.hid_size)
            sentinel_gate = layers.sigmoid(p_word_emb + p_hidden)  # [batch, hidden]
            sentinel = layers.elementwise_mul(sentinel_gate, layers.tanh(c))  # [batch, hidden]

            layers.assign(h, hid)
            layers.assign(c, cell)

            k = layers.shape(p_img_feat)[1]

            p_hid = layers.fc(h, self.hid_size, act='tanh')
            # attention 部分
            #     alpha
            hid_emb = layers.fc(p_hid, self.hid_size)  # [batch, hidden]
            exp_hid_emb = layers.expand(layers.unsqueeze(hid_emb, 1), [1, k + 1, 1])  # [batch, k+1, hidden]
            sentinel_emb = layers.unsqueeze(layers.fc(sentinel, self.hid_size), axes=1)  # [batch, 1, hidden]
            feat_emb = layers.concat([img_feat_emb, sentinel_emb], axis=1)  # [batch, k+1, hidden]
            z = layers.tanh(feat_emb + exp_hid_emb)  # [batch, k+1, 1]
            alpha = layers.fc(z, size=1, num_flatten_dims=2, act='softmax')  # [batch, k+1, 1]

            #     context vector

            context = layers.concat([img_feat, layers.unsqueeze(sentinel, axes=1)], axis=1)  # [batch, k+1, hidden]
            context = layers.elementwise_mul(context, alpha, axis=0)
            context = layers.reduce_mean(context, dim=1)  # [batch, hidden]

            out = layers.fc(context + p_hid, self.hid_size, act='tanh')

            word_pred = weight_tying_fc(out)  # [batch, vocab]

            if self.mode == 'eval':
                next_word = layers.argmax(word_pred, axis=-1)
                layers.assign(next_word, word)
                next_word = layers.cast(next_word, 'float32')
                layers.array_write(next_word, step, array=while_op_output)
            else:
                layers.array_write(word_pred, step, array=while_op_output)
            layers.increment(step)
            layers.less_than(step, max_step, cond=cond)
        if self.mode == 'train':
            output_time_major, _ = layers.tensor_array_to_tensor(while_op_output, axis=0, use_stack=True)
            output = layers.transpose(output_time_major, [1, 0, 2])
        else:
            output_time_major = layers.tensor_array_to_tensor(while_op_output, axis=0, use_stack=True)[0]
            output = layers.transpose(output_time_major, [1, 0])

        return output


class ImageCaptionModel:

    def __init__(self):
        self.encoder = MobileNetV2(trainable=encoder_config['encoder_trainable'], use_pooling=False)

    def build_input(self, mode='train'):
        if mode not in ['train', 'eval']:
            raise ValueError('不支持{}'.format(mode))
        img = fluid.data('image', [-1, 3, 224, 224])

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
        # build caption and mask
        target = caption[:, 1:]
        source = caption[:, :-1]
        padding_filled = layers.fill_constant_batch_size_like(target, shape=[-1, decoder_config['sentence_length'] - 1],
                                                              dtype='int64', value=config.dc['padding_idx'])
        mask = layers.equal(target, padding_filled)
        mask = layers.cast(layers.logical_not(mask), 'float32')
        scale_factor = layers.reduce_sum(mask)
        mask.stop_gradient = True
        scale_factor.stop_gradient = True

        # mdl
        decoder = Decoder(decoder_config['hidden_dim'], rnn_layer=1)
        image_embed, global_image_feat = self._img2feature(img)  # [batch, k+1, hidden], [batch, hidden]

        # 这里要改，要么在rnn里面做embedding，要么在外面做！
        seq_out = decoder.call(global_image_feat, image_embed, embedding_function, words=source)

        loss = layers.squeeze(ImageCaptionModel.loss(target, seq_out), axes=[2])
        loss = layers.elementwise_mul(loss, mask)
        output_loss = layers.elementwise_div(layers.reduce_sum(loss), scale_factor, name='loss')
        return output_loss

    def eval_network(self, img):
        image_embed, global_image_feat = self._img2feature(img)  # [batch, k+1, hidden], [batch, hidden]
        decoder = Decoder(decoder_config['hidden_dim'], rnn_layer=1, mode='eval')
        result = decoder.call(global_image_feat, image_embed, embedding_function)
        return result

    def _img2feature(self, img):
        img = self.encoder.net(img)
        img = layers.cast(img, 'float32')
        img = layers.reshape(img, [0, 0, -1])
        img = layers.transpose(img, [0, 2, 1])  # [batch, k, channel]
        img_embed = layers.fc(img, decoder_config['hidden_dim'], num_flatten_dims=2, act='relu')
        img_global = layers.reduce_mean(img, dim=1)  # [batch, k]
        img_global = layers.fc(img_global, decoder_config['hidden_dim'], act='relu')
        return img_embed, img_global

    @staticmethod
    def first_init(places):
        pass

    @staticmethod
    def loss(ground_true, prediction):
        # ground_true: [batch_size, seq_len]
        # prediction: [batch_size, seq_len, vocab_size]
        ground_true = layers.unsqueeze(ground_true, axes=2)
        ground_true.stop_gradient = True
        los = layers.softmax_with_cross_entropy(prediction, ground_true, axis=-1)
        return los
