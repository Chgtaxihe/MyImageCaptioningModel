from paddle import fluid
import numpy as np
import os
from PIL import Image

import config
from tools import hdf5_manager

_image_mean = np.array(config.dc['ImageMean'], dtype='float32').reshape((3, 1, 1))
_image_std = np.array(config.dc['ImageStd'], dtype='float32').reshape((3, 1, 1))


def process_image(img):
    if not isinstance(img, Image.Image):
        raise ValueError('image 应当是Image类型，而传入的是{}'.format(type(img)))
    shape = config.dc['ImageShape']
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img, dtype='float32')
    if len(img.shape) != 3:
        return None
    img = img.transpose((2, 0, 1)) / 255
    img -= _image_mean
    img /= _image_std
    return img


def read_image(path):
    return Image.open(path)


class DataReader:
    _word2index = None
    _index2word = None
    _hdf5 = None

    def init_hdf5(self):
        DataReader._hdf5 = hdf5_manager.Hdf5Manager()
        DataReader._hdf5.load_database(config.data['H5Path'])
        DataReader._hdf5.load_name2idx(config.dc['H5Name2Idx'])

    def get_reader(self, batch_size=None, mode='train'):
        if DataReader._hdf5 is None:
            self.init_hdf5()

        def h5_reader(x):
            img = DataReader._hdf5.read(x)
            return img.astype('float32')

        return self._get_reader(h5_reader, batch_size, mode)

    def _get_reader(self, processor, batch_size=None, mode='train'):
        if mode not in ['train', 'dev', 'test']:
            raise ValueError('DataReader不支持 {} 模式'.format(mode))

        if mode == 'train':
            captions, sentence_len = \
                np.load(os.path.join(config.dc['DictPath'], 'train_cap.npy'), allow_pickle=True)
        else:
            path = os.path.join(config.dc['DictPath'], 'dev_data.npy' if mode == 'dev' else 'eval_data.npy')
            files, files2cap = np.load(path, allow_pickle=True)

        def reader_train():
            for name, cap in captions:
                img = processor(name)
                caption = np.array(cap, dtype='int64')
                yield img, caption

        def reader_infer():
            for name in files:
                img = processor(name)
                caption = files2cap[name]
                yield img, caption
        rd = reader_train if mode == 'train' else reader_infer
        rd = rd if batch_size is None else fluid.io.batch(rd, batch_size)
        rd = rd if mode == 'train' else fluid.io.buffered(rd, config.train['data_loader_capacity'])
        return rd

    @property
    def word_index(self):
        if DataReader._word2index is None:
            word_index, index_word = \
                np.load(os.path.join(config.dc['DictPath'], 'word_dict.npy'), allow_pickle=True)
            DataReader._word2index = word_index
            DataReader._index2word = index_word
        return DataReader._word2index

    @property
    def index_word(self):
        if DataReader._index2word is None:
            word_index, index_word = \
                np.load(os.path.join(config.dc['DictPath'], 'word_dict.npy'), allow_pickle=True)
            DataReader._word2index = word_index
            DataReader._index2word = index_word
        return DataReader._index2word


if __name__ == "__main__":
    dr = DataReader()
    print(len(dr.index_word))
