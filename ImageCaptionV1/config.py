
data = {
    'ImagePath': r'/home/aistudio/flickr30k',  # 图片存放的目录
    'ImageFeaturePath': r'/home/aistudio/flickr30k_feature',  # 图片特征存放目录
    'ImageShape': [224, 224],  # reshape的目标大小
    'ImageMean': [0, 0, 0],  # 图片(除以255后)的均值
    'ImageStd': [1, 1, 1],  # 图片(除以255后)的标准差

    'DictPath': r'/home/aistudio/flickr30k_dict',  # caption存放的位置

    'start_idx': 2,  # `<start>` 的id
    'stop_idx': 3,  # `<stop>` 的id
    'padding_idx': 0,  # `<pad>` 的id

    'PretrainedMobileNetPath': r'/home/aistudio/MobileNetV2_pretrained',
}

train = {
    'learning_rate': 0.0001,
    'batch_size': 64,
    'data_loader_capacity': 64,
    'shuffle': True,
    'use_raw_image': False,
    'checkpoint_path': r'/home/aistudio/save',
    'checkpoint_backup_every_n_epoch': 5,
    'export_params': True,
    'export_infer_model': False,
    'max_epoch': 300,
    'log_every_n_step': 100,
    'calc_meteor': True,
    'debug_image': r'/home/aistudio/flickr30k_feature/5233757883.jpg.npy'
}

model = {
    'encoder': {
        'encoder_trainable': False,
        'encoder_dim': 7 * 7,  # 不可调
        'encoder_channel': 1280,  # 不可调
    },
    'decoder': {
        'vocab_size': 18211,  # 字典大小
        'embedding_size': 512,
        'sentence_length': 44,  # 训练用句子的最长长度(包括<start>和<end>)
        'guiding_size': 512,
        'hidden_dim': 1024,
        'enhancement': {
        },
        'use_cudnn': False,
        'infer_max_length': 44,
    }
}

log = {
    'log_path': r'/home/aistudio/log',
}

dc = data
md = model
