import numpy as np
import os
from paddle import fluid

import config
import reader as rd
from tools import util
from model.MobileNetV2 import MobileNetV2

model_name = 'show_and_tell'

info = {'show_and_tell': {
    'encoder_use_pooling': True,
    'squeeze': True
}, 'base': {
    'squeeze': 0
}
}


def get_reader():
    path = config.dc['ImagePath']
    names = [name for name in os.listdir(path) if name.endswith('jpg')]
    files = [os.path.join(path, name) for name in names]

    def reader():
        for name, f in zip(names, files):
            img = rd.process_image(rd.read_image(f))
            yield name, img

    return reader


def main(export_path=''):
    cfg = info[model_name]
    x = fluid.data('image', [None, 3, 224, 224])
    model = MobileNetV2(use_pooling=cfg.get('encoder_use_pooling', False))
    feature = model.net(x)

    cuda = fluid.CUDAPlace(0)
    exe = fluid.Executor(cuda)
    exe.run(fluid.default_startup_program())

    fluid.io.load_vars(exe, config.dc['PretrainedMobileNetPath'], fluid.default_main_program(),
                       predicate=util.get_predicate(config.dc['PretrainedMobileNetPath']))

    reader = get_reader()
    squeeze = cfg.get('squeeze', None)
    for idx, (name, img) in enumerate(reader()):
        result = exe.run(fluid.default_main_program(), feed={'image': np.expand_dims(img, 0)}, fetch_list=[feature])[0]
        if squeeze is not None:
            if squeeze is True:
                result = np.squeeze(result)
            else:
                result = np.squeeze(axis=squeeze)
        np.save(os.path.join(export_path, name), result, allow_pickle=True)
        if idx % 100 == 0:
            print('已提取{}张图片特征'.format((idx + 1)))


if __name__ == '__main__':
    if not os.path.exists(config.data['ImageFeaturePath']):
        os.makedirs(config.data['ImageFeaturePath'])
    main(config.data['ImageFeaturePath'])
