import numpy as np
import os.path as path
from PIL import Image
from paddle import fluid

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
_img_mean = np.array(_mean).reshape((3, 1, 1))
_img_std = np.array(_std).reshape((3, 1, 1))

class FlickerReader:

    def __init__(self, dirname):
        train_cap_path = path.join(dirname, 'train_cap.npy')
        dev_data_path = path.join(dirname, 'dev_data.npy')
        test_data_path = path.join(dirname, 'eval_data.npy')

        captions, word_index, index_word, sentence_len = np.load(train_cap_path, allow_pickle=True)
        self.training_data = captions
        self.word_index = word_index
        self.index_word = index_word
        self.sentence_len = sentence_len

        dev_files, dev_f2c = np.load(dev_data_path, allow_pickle=True)
        self.dev_files = dev_files
        self.dev_f2c = dev_f2c

        test_files, test_f2c = np.load(test_data_path, allow_pickle=True)
        self.test_files = test_files
        self.test_f2c = test_f2c

    def get_training_reader(self, base_dir=None, shape=(224, 224)):
        def reader():
            for file_name, caption in self.training_data:
                file_name = path.join(base_dir, file_name) if base_dir is not None else file_name
                image = self._read_image_norm(file_name, shape=shape)
                caption = np.array(caption, dtype=np.int64)
                yield image, caption
        return reader

    def get_dev_reader(self, base_dir=None, shape=(224, 224), batch_size=16):
        def reader():
            for file_name in self.dev_files:
                caption = self.dev_f2c[file_name]
                file_name = path.join(base_dir, file_name) if base_dir is not None else file_name
                image = self._read_image_norm(file_name, shape=shape)
                yield image, caption
        return fluid.io.batch(reader, batch_size, drop_last=True)

    def get_test_reader(self, base_dir=None, shape=(224, 224), batch_size=16):
        def reader():
            for file_name in self.test_files:
                caption = self.test_f2c[file_name]
                file_name = path.join(base_dir, file_name) if base_dir is not None else file_name
                image = self._read_image_norm(file_name, shape=shape)
                yield image, caption
        return fluid.io.batch(reader, batch_size, drop_last=True)
    
    @staticmethod
    def _bypass_read_image(path):
        path = path + ".npy"
        img = np.squeeze(np.load(path, allow_pickle=True), axis=0)
        return img

    @staticmethod
    def _read_image_norm(path, shape=[224, 224]):
        return FlickerReader._bypass_read_image(path)

        image = Image.open(path)
        image = image.resize(shape,Image.ANTIALIAS) if shape is not None else image
        image = np.array(image).astype(np.float32).transpose([2, 0, 1])
        image = (image / 255.) - _img_mean
        image /= _img_std
        return image.astype(np.float32)

if __name__ == "__main__":
    reader = FlickerReader(r'')
    rd = reader.get_training_reader(r'F:\Dataset\flickr8k', shape=(224, 224))
    for i in rd():
        print(i[1].shape)
        break
