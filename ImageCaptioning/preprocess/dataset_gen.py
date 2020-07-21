import json
import os
import pkuseg
import random

import config

TEMP_PATH = os.path.join(config.build_dataset['OutputPath'], 'temp')


def word_seg():
    JSON_FILE = config.build_dataset['AnnotationPath']

    length_limit = config.build_dataset['sentence_len_limit']

    if not os.path.isdir(TEMP_PATH):
        os.makedirs(TEMP_PATH, exist_ok=True)

    seg = pkuseg.pkuseg()
    used_file = []

    with open(os.path.join(TEMP_PATH, 'token'), 'w', encoding='utf-8') as o:
        with open(JSON_FILE, 'r', encoding='utf-8') as rd:
            js = json.load(rd)
            for idx, info in enumerate(js):
                name = info['image_id']
                used = False
                for i, cap in enumerate(info['caption']):
                    anno = cap.replace("\r", "").replace("\n", "").replace("、", "")
                    cut = seg.cut(anno)
                    if len(cut) > length_limit:
                        continue
                    used = True
                    anno = ' '.join(cut)
                    o.write('{}#{}\t{}\n'.format(name, i, anno))
                if used:
                    used_file.append(name)
                if idx > 0 and idx % 20000 == 0:
                    print("分词中 {}/{}".format(idx, len(js)))
    print(len(used_file))
    with open(os.path.join(TEMP_PATH, 'token.list'), 'w', encoding='utf-8') as f:
        for name in used_file:
            f.write(name + '\n')


def split_data():
    names = set()
    with open(os.path.join(TEMP_PATH, "token"), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            name = line.split("#")[0]
            names.add(name)

    names = list(names)
    random.shuffle(names)

    test_len = int(0.05 * len(names))
    val_len = int(0.05 * len(names))

    test_file = names[:test_len]
    val_file = names[test_len: val_len + test_len]

    train_file = names[test_len + val_len:]
    content = [test_file, val_file, train_file]
    filename = ["test.txt", "dev.txt", "train.txt"]
    for idx, n in enumerate(filename):
        with open(os.path.join(TEMP_PATH, n), 'w', encoding="utf-8") as f:
            for name in content[idx]:
                f.write(name + '\n')

    print("测试集大小:{}, 验证集大小:{}, 训练集大小:{}".format(test_len, val_len, len(names) - test_len - val_len))


if __name__ == '__main__':
    import tools.hdf5_manager as h5
    h5.gen_hdf5()
    word_seg()
    split_data()
    import preprocess.ai_challenge_tokenizer as tok
    tok.main()

# 运行时间: 6644.821190834045s
