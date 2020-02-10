import os
import numpy as np

import config
import reader
TARGET_PATH = config.data['ImageDecodedPath']
SOURCE_PATH = config.data['ImagePath']

file_names = [name for name in os.listdir(SOURCE_PATH) if name.endswith('jpg')]

os.makedirs(TARGET_PATH, exist_ok=True)

with open('./error.txt', 'w', encoding='utf-8') as f:
    for name in file_names:
        p = os.path.join(SOURCE_PATH, name)
        img = reader.process_image(reader.read_image(p))
        if img is None:
            f.write(name + '\n')
        else:
            np.save(os.path.join(TARGET_PATH, name), img, allow_pickle=True)

if __name__ == '__main__':
    pass
