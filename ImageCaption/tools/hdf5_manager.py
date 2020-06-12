import json
import numpy as np
import os
import re
import h5py

_split_file_pattern = re.compile(r'.*?\.hdf5_[0-9]+$')
_db_name_filter = re.compile(r'(.*?)\.hdf5[_0-9]*$')
_db_index_filter = re.compile(r'.*?\.hdf5_([0-9]+)$')
_use_float16 = True


class Hdf5Manager:

    def __init__(self):
        self._db_files = []
        self._db_lens = []
        self._name2idx = None

    def load_name2idx(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self._name2idx = json.load(f)

    def load_database(self, db_path):
        """载入数据库

        :param db_path: 数据库的目录(str)
        """
        self.close()
        if not isinstance(db_path, list):
            db_path = [db_path]

        dbs = []
        for dbp in db_path:
            files = os.listdir(dbp)

            db = [name for name in files if _split_file_pattern.match(name) is not None]
            if len(db) == 0:
                raise Exception('{} 下未找到数据库'.format(db_path))
            assert_name = _db_name_filter.findall(db[0])[0]
            is_names_equal = map(lambda x: _db_name_filter.findall(x)[0] == assert_name, db)
            if not all(is_names_equal):
                raise Exception('{} 目录下存在多个数据库'.format(db_path))
            dbs.extend(map(lambda x: os.path.join(dbp, x), db))

        if len(dbs) > 1:
            dbs.sort(key=lambda x: int(_db_index_filter.findall(x)[0]))
            print('读入数据库:\n{}'.format('\n'.join(dbs)))
        for path in dbs:
            hdf5_file = h5py.File(path, mode='r')
            self._db_files.append(hdf5_file)
            self._db_lens.append(hdf5_file['data'].shape[0])

        for i in range(1, len(self._db_lens)):
            self._db_lens[i] += self._db_lens[i-1]

    def _read(self, index):
        for idx, file in enumerate(self._db_files):
            if self._db_lens[idx] <= index:
                continue
            if idx != 0:
                index -= self._db_lens[idx - 1]
            return file['data'][index]

    def read(self, name):
        idx = self._name2idx[name]
        return self._read(idx)

    def close(self):
        for f in self._db_files:
            f.close()
        self._db_files.clear()
        self._db_lens.clear()

    def __del__(self):
        self.close()


class DbBuilder:

    def __init__(self, output_path, name, shape, max_size, db_length, dtype='float32'):
        """为3维数据创建数据库
        :param max_size: 单个文件最多放多少张图片
        :param db_length: 数据库总大小
        """
        if isinstance(shape, list):
            shape = tuple(shape)
        # assert len(shape) == 3
        self.output_path = output_path
        self.name = name
        self.shape = shape
        self.max_length = max_size
        self.db_length = db_length
        self.dtype = dtype
        self.file_index = 0
        self.ptr_index = 0
        self.cur_file = None
        self.cur_db = None

    def __enter__(self):
        if self.max_length >= self.db_length:
            p = os.path.join(self.output_path, self.name + '.hdf5')
        else:
            p = os.path.join(self.output_path, self.name + '.hdf5_{}'.format(self.file_index))
            self.file_index += 1
        self._create_new_file(p, min(self.max_length, self.db_length))

    def _create_new_file(self, path, length):
        if self.cur_file is not None:
            self.cur_file.close()
        self.cur_file = h5py.File(path, 'w')
        # gzip压缩等级并不会影响解压速度！
        self.cur_db = self.cur_file.create_dataset(name='data',
                                                   shape=[length] + list(self.shape),
                                                   dtype=self.dtype,
                                                   chunks=tuple([1] + list(self.shape)),
                                                   compression='gzip',
                                                   compression_opts=9)

    def append(self, data):
        assert np.shape(data) == self.shape
        if self.ptr_index >= self.max_length:
            p = os.path.join(self.output_path, self.name + '.hdf5_{}'.format(self.file_index))
            self.file_index += 1
            self.ptr_index = 0
            self.db_length -= self.max_length
            self._create_new_file(p, min(self.max_length, self.db_length))
        self.cur_db[self.ptr_index] = data
        self.ptr_index += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cur_file is not None:
            self.cur_file.close()


def gen_hdf5():
    import time
    begin_time = time.time()
    import reader
    image_paths = [r'F:\aichallenge\caption_train_images_20170902',
                   r'F:\aichallenge\flickr8k']
    output_path = r'F:\Dataset\challenge_flickr_mix_db'
    images = []
    name2idx = {}
    for image_path in image_paths:
        names = os.listdir(image_path)
        print(image_path, len(names))
        names = set([name for name in names if name.endswith('.jpg')])
        names = [(name, os.path.join(image_path, name)) for name in names]
        images = images + names
    builder = DbBuilder(output_path, 'aic_flk', shape=(3, 224, 224), max_size=30000, db_length=len(images),
                        dtype='float16' if _use_float16 else 'float32')
    with builder:
        for idx, (name, p) in enumerate(images):
            name2idx[name] = idx
            feat = reader.process_image(reader.read_image(p))
            if _use_float16:
                feat = feat.astype('float16')
            builder.append(feat)
            if idx % 1000 == 0:
                print(idx)

    json.dump(name2idx, open(os.path.join(output_path, 'name2idx.json'), 'w'))
    end_time = time.time()
    print("运行时间: {}s".format(end_time - begin_time))

def self_test():
    import numpy
    hdf5 = Hdf5Manager()
    hdf5.load_database(['/home/aistudio/data/data28324', '/home/aistudio/data/data28236'])
    hdf5.load_name2idx('/home/aistudio/work/ai_challenge_dict/name2idx.json')
    arr = hdf5.read('b654dd5416c1bc15369564ef622b3e0abadac3ad.jpg')
    print(arr.shape)


if __name__ == '__main__':
    gen_hdf5()
    # self_test()

