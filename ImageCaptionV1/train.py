import numpy as np
import os
import time
from paddle import fluid

import config
import evaluate
import util
from logger import Logger
from model.BaseModel import ImageCaptionModel
from reader import DataReader


decoder_config = config.md['decoder']
encoder_config = config.md['encoder']
batch_size = config.train['batch_size']
shuffle = config.train['shuffle']


def get_optimizer():
    lr = config.train['learning_rate']
    return fluid.optimizer.Adam(lr)


def training_net(use_raw_image=False):
    startup_prog, train_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            if use_raw_image:
                img = fluid.data('image', [-1, 3, 224, 224])
            else:
                img = fluid.data('image', [-1, encoder_config['encoder_channel'], 7, 7])
            caption = fluid.data('caption', [-1, decoder_config['sentence_length']], dtype='int64')

            model = ImageCaptionModel(use_raw_image)
            loss = model.training_network(img, caption)
            opt = get_optimizer()
            opt.minimize(loss)

            loader = fluid.io.DataLoader.from_generator(feed_list=[img, caption],
                                                        capacity=config.train['data_loader_capacity'])
    return loss, loader, startup_prog, train_prog


def eval_net(use_raw_image=False):
    startup_prog, eval_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            if use_raw_image:
                img = fluid.data('image', [-1, 3, 224, 224])
            else:
                # TODO
                # img = fluid.data('image', [-1, encoder_config['encoder_channel'], 7, 7])
                # 被迫使用layers.data,这里的shape检测好像有问题
                # 见本文件141行
                img = fluid.layers.data('image', [-1, encoder_config['encoder_channel'], 7, 7])

            model = ImageCaptionModel(use_raw_image)
            caption = model.eval_network(img)

    return caption, startup_prog, eval_prog


def init_data_loader(loader, places, mode='train', use_raw_image=False):
    # TODO 支持 mode = 'dev'
    if mode not in ['train']:
        raise ValueError('DataLoader不支持 {} 模式'.format(mode))

    reader = DataReader()
    if use_raw_image:
        reader = reader.get_image_reader(batch_size, mode, shuffle)
    else:
        reader = reader.get_feature_reader(batch_size, mode, shuffle)
    loader.set_sample_list_generator(reader, places)


def save_model(exe, train_program, eval_program, epoch, eval_prog_target=None):
    if eval_prog_target is not None and not isinstance(eval_prog_target, list):
        raise ValueError('eval_prog_target 应当为 None 或 List类型')

    p = config.train['checkpoint_path']
    fluid.io.save_persistables(exe, os.path.join(p, 'checkpoint'), train_program)
    n = config.train['checkpoint_backup_every_n_epoch']
    if n is not None and epoch % n == 0:
        fluid.io.save_persistables(exe, os.path.join(p, 'checkpoint{}'.format(epoch)), train_program)

    if config.train['export_params']:
        fluid.io.save_params(exe, os.path.join(p, 'params'), train_program)

    if config.train['export_infer_model'] and eval_prog_target is not None:
        fluid.io.save_inference_model(os.path.join(p, 'infer'), ['image'], eval_prog_target, exe, eval_program)


def train():
    loss, train_loader, train_startup, train_prog = training_net(config.train['use_raw_image'])
    caption, eval_startup, eval_prog = eval_net(config.train['use_raw_image'])

    places = fluid.CUDAPlace(0)
    exe = fluid.Executor(places)
    exe.run(train_startup)
    exe.run(eval_startup)

    # 填充数据
    init_data_loader(train_loader, places, 'train', config.train['use_raw_image'])

    # 加载检查点(如果有)
    logger = Logger()
    if logger.is_first_init:
        if config.train['use_raw_image']:  # 加载MobileNetV2参数
            fluid.io.load_vars(exe, config.dc['PretrainedMobileNetPath'], train_prog,
                               predicate=util.get_predicate(config.dc['PretrainedMobileNetPath'], warning=False))
    else:  # 恢复上次训练的进度
        fluid.io.load_persistables(exe, os.path.join(config.train['checkpoint_path'], 'checkpoint'), train_prog)

    for epoch in range(logger.epoch, config.train['max_epoch'] + 1):
        begin_time = time.time()
        logger.log("Epoch {}".format(epoch))
        epoch_loss = 0
        for step, data in enumerate(train_loader()):
            step_loss = exe.run(train_prog, feed=data, fetch_list=[loss])[0]
            if np.isnan(step_loss).any():  # 检查Nan
                raise AssertionError('Epoch:{} Step:{} Loss为Nan'.format(epoch, step + 1))
            epoch_loss += step_loss[0]

            # log
            if (step + 1) % config.train['log_every_n_step'] == 0:
                logger.log(' ' * 4 + 'Step {} Mean loss: {:6f} Step loss: {:6f}'.
                          format(step + 1, epoch_loss / (step + 1) / batch_size, step_loss[0] / batch_size))
        logger.log('Epoch loss: {:7f}'.format(epoch_loss / (step + 1) / batch_size))

        # 保存模型
        save_model(exe, train_prog, eval_prog, epoch, [caption])

        # 计算验证集成绩
        dr = DataReader()
        dr = dr.get_image_reader(batch_size, 'dev', shuffle) if config.train['use_raw_image'] else \
            dr.get_feature_reader(batch_size, 'dev', shuffle)
        bleu_score, meteor_score = 0, 0
        calc_meteor = config.train['calc_meteor']
        for l, data in enumerate(dr()):
            img, real_cap = zip(*data)
            # TODO 这里怎么回事？？？
            # img = np.reshape(img, [-1, 1280, 49])
            cp = exe.run(eval_prog, feed={'image': img}, fetch_list=[caption])[0]
            bleu_score += evaluate.calc_bleu(cp, real_cap)
            if calc_meteor:
                meteor_score += evaluate.calc_meteor(cp, real_cap)
        bleu_score /= l + 1
        meteor_score /= l + 1
        logger.log('Dev set: BLEU 分数: {:.7f} Meteor 分数: {:.7f}'.format(bleu_score, meteor_score))
        if config.train['debug_image'] is not None:
            i = np.load(config.train['debug_image'], allow_pickle=True)
            t = exe.run(eval_prog, feed={'image': np.expand_dims(i, 0)}, fetch_list=[caption])[0]
            logger.log('测试输出:{}'.format(evaluate.words2sentence(evaluate.filter(t[0]))))
        
        logger.log('Epoch 耗时 {:2f}s'.format(time.time() - begin_time))
        
        logger.epoch = epoch


if __name__ == '__main__':
    train()
