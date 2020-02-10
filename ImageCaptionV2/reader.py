from paddle import fluid
import numpy as np
import os
import h5py
import json
from PIL import Image

import config
from tools import multiprocess

_image_mean = np.array(config.dc['ImageMean'], dtype='float32').reshape((3, 1, 1))
_image_std = np.array(config.dc['ImageStd'], dtype='float32').reshape((3, 1, 1))

use_raw_image = config.train['use_raw_image']
use_hdf5 = config.train['use_h5py']
use_decoded_img = config.train['use_decoded_image']
use_multiprocess_reader = config.train['use_multiprocess_reader']


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
    multiprocess_helper = None

    def __init__(self):
        self.name2idx = None
        self.hdf5 = None
        self.hdf5_imgs = None

    def stop(self):
        if DataReader.multiprocess_helper is not None:
            DataReader.multiprocess_helper.stop_immediate()
        if self.hdf5 is not None:
            self.hdf5.close()

    def __del__(self):
        self.stop()

    def init_hdf5(self):
        if use_hdf5 and self.name2idx is None:
            self.name2idx = json.load(open(config.dc['H5Name2Idx'], 'r'))
            self.hdf5 = h5py.File(os.path.join(config.dc['H5Path']), 'r')
            self.hdf5_imgs = self.hdf5['image']

    def get_reader(self, batch_size=None, mode='train'):
        if mode == 'train' and use_multiprocess_reader:
            assert use_raw_image and use_decoded_img
            return self._get_reader_multiprocess(batch_size, mode)
        if use_hdf5:
            self.init_hdf5()
            return self._get_reader(lambda x: self.hdf5_imgs[self.name2idx[x], :, :, :], batch_size, mode)
        if use_raw_image:
            if use_decoded_img:
                base_path = config.dc['ImageDecodedPath']
                return self._get_reader(lambda x: np.load(os.path.join(base_path, x+'.npy'), allow_pickle=True),
                                        batch_size, mode)
            else:
                base_path = config.dc['ImagePath']
                return self._get_reader(lambda x: process_image(read_image(x)), batch_size, mode)
        else:
            base_path = config.dc['ImageFeaturePath']
            return self._get_reader(lambda x: np.load(os.path.join(base_path, x+'.npy'), allow_pickle=True),
                                    batch_size, mode)

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

    def _get_reader_multiprocess(self, batch_size=None, mode='train'):
        if mode not in ['train']:
            raise ValueError('{} 暂不支持'.format(mode))
        captions, sentence_len = \
            np.load(os.path.join(config.dc['DictPath'], 'train_cap.npy'), allow_pickle=True)
        if DataReader.multiprocess_helper is None and use_multiprocess_reader:
            def reader(file_path, caption):
                return np.load(file_path, allow_pickle=True), caption

            DataReader.multiprocess_helper = multiprocess.MultiProcessReader()
            DataReader.multiprocess_helper.start(max_processes=8, work_function=reader)

        def multiprocess_reader():
            base_dir = config.data['ImageDecodedPath']
            total_len = len(captions)
            output_cnt = 0
            for name, cap in captions:
                p = os.path.join(base_dir, name)+'.npy'
                self.multiprocess_helper.add_request(file_path=p,
                                                     caption=cap)
            while output_cnt < total_len:
                yield self.multiprocess_helper.get_result()
                output_cnt += 1
        rd = multiprocess_reader
        return rd if batch_size is None else fluid.io.batch(rd, batch_size)

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
