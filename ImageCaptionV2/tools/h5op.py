"""
    该文件用于创建HDF5数据集
"""
import json
import h5py
import config
import numpy as np
import os

use_raw = config.train['use_raw_image']
use_float16 = True

def gen_hdf5():
    assert use_raw
    names = os.listdir(r'F:\flickr30k_decoded')
    names = set([name[:-4] for name in names if name.endswith('.npy')])
    image_path = r'F:\flickr30k_decoded'
    h5 = h5py.File(r'F:\train_data.hdf5', 'w')
    # gzip压缩等级并不会影响解压速度！
    img_db = h5.create_dataset('image',
                               shape=(len(names), 3, 224, 224),
                               dtype='float32' if not use_float16 else 'float16',
                               chunks=(1, 3, 224, 224),
                               compression='gzip',
                               compression_opts=9)
    name2idx = {}
    json.dump(name2idx, open(os.path.join(config.dc['DictPath'], 'name2idx.json'), 'w'))
    for idx, name in enumerate(names):
        name2idx[name] = idx
        name = name + '.npy'
        feat = np.load(os.path.join(image_path, name), allow_pickle=True)
        if use_float16:
            feat = feat.astype('float16')
        img_db[idx, :, :, :] = feat
        if idx % 1000 == 0:
            print(idx)
    h5.close()
    json.dump(name2idx, open(os.path.join(config.dc['DictPath'], 'name2idx.json'), 'w'))


if __name__ == '__main__':
    gen_hdf5()
