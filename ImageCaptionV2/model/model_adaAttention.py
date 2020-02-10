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
    return layers.elementwise_add(layers.matmul(x, embedding_w, transpose_y=True), bias)


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
        step = layers.fill_constant(shape=[1], dtype='int64', value=0)
        mx = decoder_config['sentence_length'] - 1
        if self.mode == 'eval':
            mx = decoder_config['infer_max_length']
            while_op_output = layers.create_array('int64')
        else:
            while_op_output = layers.create_array('float32')
        max_step = layers.fill_constant(shape=[1], dtype='int64', value=mx)
        cond = layers.less_than(step, max_step)
        while_op = layers.While(cond)

        with while_op.block():
            if self.mode == 'train':
                s = layers.cast(step, 'int32')
                word = layers.slice(words, axes=[0], starts=s, ends=s + 1)
                word = layers.squeeze(word, [0])
                word.stop_gradient = True
                # 估计是paddle的bug，查了好久才找到解决方法
                # https://github.com/PaddlePaddle/Paddle/issues/21999

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
                layers.array_write(next_word, step, array=while_op_output)
            else:
                layers.array_write(word_pred, step, array=while_op_output)
            # 更新while条件
            layers.increment(step, in_place=True)
            layers.less_than(step, max_step, cond=cond)

        output_time_major, _ = layers.tensor_array_to_tensor(while_op_output, axis=0, use_stack=True)
        if self.mode == 'eval':
            output = layers.transpose(output_time_major, [1, 0])
        else:
            output = layers.transpose(output_time_major, [1, 0, 2])
        return output


class ImageCaptionModel:

    def __init__(self, use_raw_image=False):
        self.use_raw_image = use_raw_image
        if use_raw_image:
            self.encoder = MobileNetV2(trainable=encoder_config['encoder_trainable'], use_pooling=False)

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

        return layers.reduce_sum(loss) / scale_factor

    def eval_network(self, img):
        image_embed, global_image_feat = self._img2feature(img)  # [batch, k+1, hidden], [batch, hidden]
        decoder = Decoder(decoder_config['hidden_dim'], rnn_layer=1, mode='eval')
        mode = decoder_config['infer_mode']
        if mode not in ['normal']:
            raise ValueError('Infer mode {} is not supported!'.format(mode))

        if mode == 'normal':
            result = decoder.call(global_image_feat, image_embed, embedding_function)

        return result

    def _img2feature(self, img):
        if self.use_raw_image:
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
        # 之所以要stop_gradient，请见，估计是个Bug
        # https://github.com/PaddlePaddle/Paddle/issues/19750
        ground_true.stop_gradient = True
        los = layers.softmax_with_cross_entropy(prediction, ground_true, axis=-1)
        return los


# 使用了Flickr30k，删除了长度大于22的句子且去除了出现次数小于3的单词（但验证集/测试集的单词没有做unk处理）
# 日志文件: https://share.dmca.gripe/weIF8u3OW5MizyRr.txt
# 图像: https://share.dmca.gripe/m7akUr4ksVmVkzeK.png
# unk_size: 6329	训练集大小:129129	字典大小:7396	句子长度:24	<start>:2	<stop>:3
# Meteor 分数: 0.2814378 (best meteor)
# BLEU [0.4782019, 0.2308212, 0.2280953, 0.1861378] 0.2808140
# 模型一共说了991句不同的话
# Meteor 分数: 0.2745945 (best bleu)
# BLEU [0.4966271, 0.2390398, 0.2266436, 0.1871630] 0.2873684
# 模型一共说了758句不同的话
# Meteor 分数: 0.2781405 (Epoch 193)
# BLEU [0.4722045, 0.2319533, 0.2268296, 0.1851085] 0.2790240
# 模型一共说了974句不同的话
# 还不如 Model Enhanced
# 不知道是不是单词表缩小的缘故，新的训练集信息如下
# unk_size: 0	训练集大小:133980	字典大小:14112	句子长度:32	<start>:2	<stop>:3
