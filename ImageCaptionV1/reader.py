from paddle import fluid
import numpy as np
import os
import random
from PIL import Image

import config

_image_mean = np.array(config.dc['ImageMean'], dtype='float32').reshape((3, 1, 1))
_image_std = np.array(config.dc['ImageStd'], dtype='float32').reshape((3, 1, 1))


def process_image(img):
    if not isinstance(img, Image.Image):
        raise ValueError('image 应当是Image类型，而传入的是{}'.format(type(img)))

    shape = config.dc['ImageShape']
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img, dtype='float32').transpose((2, 0, 1)) / 255
    img -= _image_mean
    img /= _image_std

    return img


def read_image(path):
    return Image.open(path)


class DataReader:
    _word2index = None
    _index2word = None

    '''
        获取图片特征(由CNN提取)与标注(caption)
    '''

    def get_feature_reader(self, batch_size=None, mode='train', shuffle=False):
        def processor(path):
            return np.load(path, allow_pickle=True)

        return self.get_reader(processor, batch_size, mode, shuffle)

    '''
        获取图片与标注(caption)
    '''

    def get_image_reader(self, batch_size=None, mode='train', shuffle=False):
        def processor(path):
            return process_image(read_image(path))

        return self.get_reader(processor, batch_size, mode, shuffle)

    def get_reader(self, processor, batch_size=None, mode='train', shuffle=False):
        if mode not in ['train', 'dev', 'test']:
            raise ValueError('DataReader不支持 {} 模式'.format(mode))

        if mode == 'train':
            captions, word_index, index_word, sentence_len = \
                np.load(os.path.join(config.dc['DictPath'], 'train_cap.npy'), allow_pickle=True)
            DataReader._word2index = word_index
            DataReader._index2word = index_word
            if shuffle:
                random.shuffle(captions)
        if mode == 'dev' or mode == 'test':
            path = os.path.join(config.dc['DictPath'], 'dev_data.npy' if mode == 'dev' else 'eval_data.npy')
            files, files2cap = np.load(path, allow_pickle=True)
            if shuffle:
                random.shuffle(files)

        use_raw_image = config.train['use_raw_image']
        image_dir = config.dc['ImagePath'] if use_raw_image else config.dc['ImageFeaturePath']

        def reader_train():
            for name, cap in captions:
                file_path = os.path.join(image_dir, name)
                if not use_raw_image:
                    file_path = file_path + '.npy'
                img = processor(file_path)
                caption = np.array(cap, dtype='int64')
                yield img, caption

        def reader_infer():
            for name in files:
                file_path = os.path.join(image_dir, name)
                if not use_raw_image:
                    file_path = file_path + '.npy'
                img = processor(file_path)
                caption = files2cap[name]
                yield img, caption

        rd = reader_train if mode == 'train' else reader_infer
        return rd if batch_size is None else fluid.io.batch(rd, batch_size)

    @property
    def word_index(self):
        if DataReader._word2index is None:
            _, word_index, index_word, _ = \
                np.load(os.path.join(config.dc['DictPath'], 'train_cap.npy'), allow_pickle=True)
            DataReader._word2index = word_index
            DataReader._index2word = index_word
        return DataReader._word2index

    @property
    def index_word(self):
        if DataReader._index2word is None:
            _, word_index, index_word, _ = \
                np.load(os.path.join(config.dc['DictPath'], 'train_cap.npy'), allow_pickle=True)
            DataReader._word2index = word_index
            DataReader._index2word = index_word
        return DataReader._index2word


if __name__ == "__main__":
    dr = DataReader()
    print(len(dr.index_word))
