
build_dataset = {
    'ImagePaths': [r'F:\Dataset\challenge\caption_train_images_20170902', ],
    'AnnotationPath': r'F:\Dataset\challenge\caption_train_annotations_20170902.json',
    
    'OutputPath': r'G:\Dataset\challenge',
    'H5Name2Idx': r'G:\Dataset\challenge\name2idx.json',

    'sentence_len_limit': 33,  # 句子最大长度(不含<start>/<stop>标记)
    'compression_opts': 5,  # gzip等级 (0-9)
    'max_keep': 15000  # 最多保留多少个词
}

data = {
    'ImageShape': [224, 224],  # reshape的目标大小
    'ImageMean': [0, 0, 0],  # 图片(除以255后)的均值
    'ImageStd': [1, 1, 1],  # 图片(除以255后)的标准差

    'DictPath': r'/home/aistudio/data/data45289/',  # npy文件存放的目录
    'H5Path': ['/home/aistudio/data/data45314/'],  # HDF5目录
    'H5Name2Idx': '/home/aistudio/data/data45314/name2idx.json',  # name2idx.json文件位置
    'start_idx': 2,  # `<start>` 的id
    'stop_idx': 3,  # `<stop>` 的id
    'padding_idx': 0,  # `<pad>` 的id (无需修改)

    'PretrainedMobileNetPath': None,  # 预训练的MobileNetV2模型参数 (或None)
    'sample_count': 944996  # 训练集大小
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
    'checkpoint_path': r'/home/aistudio/work/save',
    'checkpoint_backup_every_n_epoch': False,  # False 或 数字
    'export_params': False,
    'export_infer_model': True,  # 导出预测模型
    'max_epoch': 10,
    'log_every_n_step': 150,
    'save_best_bleu_checkpoint': True
}

model = {
    'encoder': {
        'encoder_trainable': True,  # 为False时,generic caption问题严重，估计是预训练模型不够好？？
        'encoder_dim': 7 * 7,  # MobileNetV2的参数，不可调
        'encoder_channel': 1280,  # MobileNetV2的参数，不可调
    },
    'decoder': {
        'vocab_size': 12295,  # 字典大小
        'embedding_size': 256,  # 使用weight tying，故必须与hidden_dim相同
        'sentence_length': 35,  # 句子长度
        # TODO 添加sentinel_size的设置
        'hidden_dim': 1024,  # lstm隐藏层大小/sentinel_size
        'infer_max_length': 35,
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
