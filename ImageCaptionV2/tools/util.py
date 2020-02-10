import math

from paddle import fluid
from paddle.fluid import layers

import config


def read_file(path, mode='r'):
    with open(path, mode=mode, encoding='utf-8') as f:
        context = f.read()
    return context


def write_file(path, context, mode='w'):
    with open(path, mode=mode, encoding='utf-8') as f:
        f.write(context)


def get_lr(strategy, base_lr, sample_cnt, batch_size, current_epoch=0):
    if strategy not in [None, 'cosine_decay', 'cosine_decay_restart']:
        raise ValueError('Lr衰减策略错误')
    step_each_epoch = math.ceil(sample_cnt / batch_size)
    if strategy == 'cosine_decay':
        return fluid.layers.cosine_decay(base_lr,
                                         step_each_epoch=step_each_epoch,
                                         epochs=config.train['decay_epoch'])
    if strategy == 'cosine_decay_restart':
        return cosine_decay_restart(base_lr,
                                    step_each_epoch=step_each_epoch,
                                    decay_epochs=config.train['decay_epoch'],
                                    current_epoch=current_epoch)
    if strategy is None:
        return fluid.layers.fill_constant([1], 'float32', base_lr)


def _decay_step_counter(begin=0):
    # 从paddle源码copy来的
    global_step = layers.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    layers.cast(global_step, 'float32')
    return global_step


def cosine_decay_restart(learning_rate, step_each_epoch, decay_epochs, m_mul=1.0, t_mul=2.0, current_epoch=0):
    """
        各参数含义请见 tensorflow api
        https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/train/cosine_decay_restarts
    """
    global_step = layers.cast(_decay_step_counter(current_epoch), 'float32')
    cur_epoch = layers.floor(global_step / step_each_epoch)
    completed_fraction = cur_epoch / decay_epochs
    if t_mul == 1.0:
        i_restart = layers.floor(completed_fraction)
        completed_fraction -= i_restart
    else:
        i_restart = layers.floor(layers.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                                 layers.log(t_mul))
        sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
        completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

    i_restart = layers.cast(i_restart, 'int64')
    m_fac = layers.cast(m_mul ** i_restart, 'float32')
    cosine_decayed = 0.5 * m_fac * (layers.cos(math.pi * completed_fraction) + 1)
    return learning_rate * cosine_decayed


def get_predicate(path, warning=True):
    """
        获取用于加载权重的predicate
    """

    def predicate(var):
        from paddle.fluid.framework import Parameter
        import os
        if not isinstance(var, Parameter): return False
        file_path = os.path.normpath(os.path.join(path, var.name))
        if not os.path.isfile(file_path):
            if warning:
                print('ERROR: %s not found!' % var.name)
            return False
        return True

    return predicate
