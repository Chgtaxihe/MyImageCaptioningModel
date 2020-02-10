
data = {
    'ImagePath': r'/home/aistudio/flickr30k',  # 图片存放的目录
    'ImageDecodedPath': r'/home/aistudio/flickr30k_decoded',
    'ImageFeaturePath': r'/home/aistudio/flickr30k_feature',  # 图片特征存放目录
    'ImageShape': [224, 224],  # reshape的目标大小
    'ImageMean': [0, 0, 0],  # 图片(除以255后)的均值
    'ImageStd': [1, 1, 1],  # 图片(除以255后)的标准差

    'DictPath': r'/home/aistudio/work/flickr30k_dict',  # caption存放的位置
    # 'DictPath': r'F:\Dataset\flickr30k_dict',
    'H5Path': '/home/aistudio/data/data20651/train_data.hdf5',
    'H5Name2Idx': '/home/aistudio/data/data20651/name2idx.json',
    'start_idx': 2,  # `<start>` 的id
    'stop_idx': 3,  # `<stop>` 的id
    'padding_idx': 0,  # `<pad>` 的id

    'PretrainedMobileNetPath': r'/home/aistudio/work/MobileNetV2_pretrained',
    'sample_count': 133980  # 训练集大小
}

train = {
    'seed': 12345,  # None 或任意数字
    'learning_rate': 0.0003,
    'lr_decay_strategy': 'cosine_decay',  # None / cosine_decay / cosine_decay_restart
    'decay_epoch': 20,  # 仅当decay策略为cosine_decay_restart时有效
    'gradient_clip': False,
    'batch_size': 128,
    'data_loader_capacity': 128,
    'use_raw_image': True,
    'use_h5py': True,
    'use_decoded_image': True,
    # 多进程只支持use_decoded_image is True且use_h5py is False
    'use_multiprocess_reader': False,  # 即便用了多进程，也快不了几秒钟，不推荐在AIstudio上使用（怕不是我代码写的太烂？）
    'checkpoint_path': r'/home/aistudio/work/save',
    'checkpoint_backup_every_n_epoch': False,
    'export_params': True,
    'export_infer_model': True,
    'max_epoch': 1500,
    'log_every_n_step': 150,
    'calc_meteor': True,
    'save_best_bleu_checkpoint': True,
    'save_best_meteor_checkpoint': True,  # 仅当 calc_meteor == True 时生效
}

model = {
    'encoder': {
        'encoder_trainable': True,
        'encoder_dim': 7 * 7,  # 不可调
        'encoder_channel': 1280,  # 不可调
    },
    'decoder': {
        'vocab_size': 14112,  # 字典大小
        'embedding_size': 512,
        'sentence_length': 32,  # 训练用句子的最长长度(包括<start>和<end>)
        'guiding_size': 512,
        'hidden_dim': 512,
        'infer_max_length': 32,
        'infer_mode': 'normal',  # 'normal' 或 'beam_search'
        'beam_size': 4,  # 仅当 infer_mode == 'beam_search' 时生效
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
