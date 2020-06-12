
data = {
    'ImageShape': [224, 224],  # reshape的目标大小
    'ImageMean': [0, 0, 0],  # 图片(除以255后)的均值
    'ImageStd': [1, 1, 1],  # 图片(除以255后)的标准差

    'DictPath': r'/home/aistudio/work/challege_flickr_dic',  # caption存放的位置
    'H5Path': ['/home/aistudio/data/data31554'],  # HDF5目录
    'H5Name2Idx': '/home/aistudio/work/challege_flickr_dic/name2idx.json',  # 图片-索引对应表
    'start_idx': 2,  # `<start>` 的id
    'stop_idx': 3,  # `<stop>` 的id
    'padding_idx': 0,  # `<pad>` 的id

    'PretrainedMobileNetPath': r'/home/aistudio/work/MobileNetV2_pretrained',  # 预训练的MobileNetV2模型参数 (可填None)
    'sample_count': 981413  # 训练集大小
}

train = {
    'seed': None,  # None 或任意数字
    'learning_rate': 0.00005,
    'lr_decay_strategy': None,  # 见tools/util.py
    'decay_epoch': 0,  # 仅当decay策略为cosine_decay_restart时有效
    'warmup_epoch': 3,
    'gradient_clip': False,
    'batch_size': 128,
    'data_loader_capacity': 128,
    'use_h5py': True,
    'use_decoded_image': True,
    'checkpoint_path': r'/home/aistudio/work/save',
    'checkpoint_backup_every_n_epoch': False,  # False 或 数字
    'export_params': False,
    'export_infer_model': True,
    'max_epoch': 10,
    'log_every_n_step': 150,
    'calc_meteor': False,
    'save_best_bleu_checkpoint': True,
    'save_best_meteor_checkpoint': False,  # 仅当 calc_meteor == True 时生效
}

model = {
    'encoder': {
        'encoder_trainable': True,  # 为False时,generic caption问题严重，估计是预训练模型不够好？？
        'encoder_dim': 7 * 7,  # MobileNetV2的参数，不可调
        'encoder_channel': 1280,  # MobileNetV2的参数，不可调
    },
    'decoder': {
        'vocab_size': 13141,  # 字典大小
        'embedding_size': 256,  # 使用weight tying，故必须与hidden_dim相同
        'sentence_length': 35,  # 训练用句子的最长长度(包括<start>和<end>)
        # TODO 添加sentinel_size的设置
        'hidden_dim': 1024,  # lstm隐藏层大小/sentinel_size
        'infer_max_length': 32,
    }
}

log = {
    'log_path': r'/home/aistudio/work/log',
}

evaluate = {
    'sentence_statistics': True
}

dc = data
md = model
