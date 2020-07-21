import json
import os
import numpy as np
import random

import config

CAPTION_PATH = os.path.join(config.build_dataset['OutputPath'], 'temp')
OUTPUT_DIR = config.build_dataset['OutputPath']
TRAINING_SPLIT = os.path.join(CAPTION_PATH, 'train.txt')
DEV_SPLIT = os.path.join(CAPTION_PATH, 'dev.txt')
EVAL_SPLIT = os.path.join(CAPTION_PATH, 'test.txt')
TOKEN_PATH = os.path.join(CAPTION_PATH, 'token')
seed = 123456789
minimum_occur = 2  # 一个单词最少要出现的次数
max_keep = config.build_dataset['max_keep']

random.seed(seed)


def build_dict(path, train_div=None):
    word_cnt = dict()
    if train_div is not None:
        train_div = set(train_div)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, line = line.split('\t')
            name = name[:-2]
            if train_div is not None:
                if name not in train_div:
                    continue
            words = list(line.strip().split())
            for word in words:
                word_cnt[word] = word_cnt.get(word, 0) + 1

    words = sorted(word_cnt.items(), key=lambda k: k[1], reverse=True)
    keep = min(max_keep, len(words))
    reduced_word, removed_word = [], []
    for idx, (word, cnt) in enumerate(words, 1):
        if cnt < minimum_occur or idx > keep:
            removed_word.append(word)
        else:
            reduced_word.append(word)
    print(removed_word[:min(5, len(removed_word))], ' 等词语被删除.')
    print('unk_size: {}'.format(len(words) - len(reduced_word)), end='\t')

    reduced_word = ['<pad>', '<unk>', '<start>', '<stop>'] + list(reduced_word)
    idx_word = {idx: word for idx, word in enumerate(reduced_word)}
    word_idx = {word: idx for idx, word in enumerate(reduced_word)}
    return word_idx, idx_word


def tokenize(path, word_idx):
    assert isinstance(word_idx, dict)
    tokens = []
    unk_idx = word_idx['<unk>']
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, context = line.split('\t')
            name = name[:-2]
            words = list(context.strip().split())
            token = [word_idx.get(w, unk_idx) for w in words]
            tokens.append((name, token))
    random.shuffle(tokens)
    return tokens


def load_division(path):
    with open(path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines()]
    return names


def build_training_db(word_idx, tokens, names, sentences_per_image=5):
    start, stop = word_idx['<start>'], word_idx['<stop>']
    cnt = {name: 0 for name in names}
    result = []
    max_len = 0
    for name, token in tokens:
        if name in cnt and cnt[name] < sentences_per_image:
            token = [start] + token + [stop]
            max_len = max(max_len, len(token))
            cnt[name] += 1
            result.append((name, np.array(token, dtype='int64')))
    result = [(name, np.pad(token, (0, max_len - len(token)), 'constant', constant_values=0))
              for name, token in result]
    random.shuffle(result)
    return result, max_len


def build_eval_db(path, names):
    result = {name: [] for name in names}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, content = line.split('\t')
            name = name[:-2]
            if name in result:
                result[name].append(list(content.strip().split()))
    return result


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    train, dev, test = map(load_division, [TRAINING_SPLIT, DEV_SPLIT, EVAL_SPLIT])
    word_idx, idx_word = build_dict(TOKEN_PATH, train)
    np.save(os.path.join(OUTPUT_DIR, 'word_dict.npy'), [word_idx, idx_word], allow_pickle=True)

    tokens = tokenize(TOKEN_PATH, word_idx)

    train_db, max_len = build_training_db(word_idx, tokens, train)
    np.save(os.path.join(OUTPUT_DIR, 'train_cap.npy'), [train_db, max_len], allow_pickle=True)

    p = TOKEN_PATH
    np.save(os.path.join(OUTPUT_DIR, 'dev_data.npy'), [dev, build_eval_db(p, dev)], allow_pickle=True)
    np.save(os.path.join(OUTPUT_DIR, 'eval_data.npy'), [test, build_eval_db(p, test)], allow_pickle=True)

    print('训练集大小:{}\t字典大小:{}\t句子长度:{}\t<start>:{}\t<stop>:{}'.
          format(len(train_db), len(word_idx), max_len, word_idx['<start>'], word_idx['<stop>']))

    with open(os.path.join(OUTPUT_DIR, "word2idx.json"), 'w', encoding='utf-8') as f:
        json.dump(word_idx, f, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "idx2word.json"), 'w', encoding='utf-8') as f:
        json.dump(idx_word, f, ensure_ascii=False)


if __name__ == '__main__':
    main()

# ['逮捕', '障碍台', '捉鱼虾', '斑点裤', '尸检']  等词语被删除.
# unk_size: 8160	训练集大小:944996	字典大小:12295	句子长度:35	<start>:2	<stop>:3
