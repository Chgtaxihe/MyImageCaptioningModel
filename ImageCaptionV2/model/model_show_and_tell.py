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
        final_output = layers.fc(output,
                                 decoder_config['vocab_size'],
                                 num_flatten_dims=2,
                                 param_attr=ParamAttr('output_fc_w'),
                                 bias_attr=ParamAttr('output_fc_b'))
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
        output_fn = lambda a: layers.fc(a,
                                        decoder_config['vocab_size'],
                                        param_attr=ParamAttr('output_fc_w'),
                                        bias_attr=ParamAttr('output_fc_b'))
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
                output = layers.fc(output,
                                   decoder_config['vocab_size'],
                                   num_flatten_dims=1,
                                   param_attr=ParamAttr('output_fc_w'),
                                   bias_attr=ParamAttr('output_fc_b'))
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
# 存在问题: generic caption
# seed: 666666666
# Meteor 分数: 0.2275348
# BLEU [0.423974, 0.221814, 0.216701, 0.187436] 0.2624811

# Epoch 1500
# lr: 1e-5, cosine decay
# Meteor 分数: 0.2332823
# BLEU [0.401945, 0.203863, 0.223936, 0.197728] 0.2568677
# seed: 12345
# 模型一共说了497句不同的话
# 启用Encoder训练，在2k的训练集上：训练200Epoch左右时，网络在TestSet上只会说一句话
# 在417Epoch(best bleu)，会说143句
# 在819Epoch(best meteor), 会说444句
# 1500Epoch，会说497句
# 可见generic caption问题的解决，依赖于一个可训练的Encoder（或是Encoder预训练不足）
# 同时应给予足够的训练时间
# 训练的结果见图 https://share.dmca.gripe/T3NTBm3KtWOuA8Ri.png
# 本次训练一直到1500Epoch未见Nan，可能的原因有：（以下实验依次进行）
# 1. 启用了Encoder的训练.
#    尝试禁用Encoder训练后，605Epoch后仍未Nan
#    BLEU [0.268510733230755, 0.15376629044559637, 0.22467409882655326, 0.20257176408180697 ] 0.2123807216461779
#    同时注意到，605Epoch时，该网络在TestSet上只说了一句话，证明Encoder训练不足
#    (或Encoder训练充足，但打开Encoder训练有益于语言多样性)
#    且从图像中可看到，BLEU抖动剧烈(可能为偶发现象，不具代表性)
#    https://share.dmca.gripe/EHewu6OkjfDixPnO.png
# 2. lstm_w 使用了GradientClipByGlobalNorm， norm=0.05
#    为了加快实验，本次实验采用1的结果，关闭Encoder训练，取消GradientClipByGlobalNorm
#    600Epoch未见Nan
#        Meteor 分数: 0.1966298
#        BLEU [0.3600043, 0.1888368, 0.2213182, 0.1961652] 0.2415811
#        模型一共说了1句不同的话
#    图像: https://share.dmca.gripe/shjuEuso1lbUzoIC.png
# 3. lstm_w 的初始化使用的是mean=0, std=0.01的正态分布
#    禁用Encoder训练，不使用GradientClip，lstm_w使用默认初始化(Xavier，均匀分布)
#    800Epoch(loss:)未见Nan
#        Meteor 分数: 0.2535998
#        BLEU [0.3151248, 0.1490877, 0.1951733, 0.1909020] 0.2125720
#        模型一共说了2句不同的话
#    更换种子: 666666, Epoch 840未见Nan
#        Meteor 分数: 0.2241075
#        BLEU [0.3285403, 0.1505138, 0.2126167, 0.2027638] 0.2236086
#        模型一共说了1句不同的话
#    图像: https://share.dmca.gripe/NINMcpuUnhE6Tlm4.png
# 4. 使用了2k图片，每张图片1句话的小训练集
#    改用30k图片，每张图片1句话（对应的vocab_size, sample_cnt亦增大），禁用Encoder，其余与初始设定一致
#    训练到 707 Epoch(Loss:2.317456)，未见Nan
#       Meteor 分数: 0.2332897
#       BLEU [0.3026205, 0.1712951, 0.1978389, 0.1886411] 0.2150989
#       模型一共说了2句不同的话
#    图像: https://share.dmca.gripe/PJmqPOkLKXXNMsKv.png
# 接着试验4，启用Encoder继续训练
#     Epoch 787
#     Meteor 分数: 0.2319043
#     BLEU [0.3341896, 0.1744932, 0.2064051, 0.1920142] 0.2267755
#     模型一共说了18句不同的话
# 重新训练，启用Encoder，使用30k图片，每张图片1句话
# seed: 12345
# 训练了483Epoch Best_BLEU在349 Best_Meteor在470
# 图像: https://share.dmca.gripe/AZF8C5Ad3mBzNBlK.png
# 注明: Test set 有3178张不同的图片
# Best BLEU:
#     Meteor 分数: 0.2631433
#     BLEU [0.4146437, 0.1867975, 0.2067277, 0.1927320] 0.2502252
#     模型一共说了1493句不同的话
# Best Meteor:
#     Meteor 分数: 0.2700730
#     BLEU [0.4051078, 0.1818795, 0.2082690, 0.1954095] 0.2476664
#     模型一共说了1565句不同的话
# 尽管如此，模型预测出来的句子仍然是惨不忍睹（图文无关）
# Epoch 1169:
#     Meteor 分数: 0.2702888
#     BLEU [0.4046879, 0.1823192, 0.2122078, 0.1959406] 0.2487889
#     模型一共说了1589句不同的话
# infer结果依旧惨不忍睹
# 图像：https://share.dmca.gripe/NEVJeQZRhWhxNrks.png
# (400-600Epoch中Loss的峰是因为重启训练，导致cosine decay重置，lr上升导致)
# 注意到Meteor有所上升，是否应该用带cycle的lr呢？
