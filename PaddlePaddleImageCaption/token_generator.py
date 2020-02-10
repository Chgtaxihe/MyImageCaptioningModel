import numpy as np
import os.path as path
import random
import tensorflow as tf
import re

CAPTION_PATH = r'/home/aistudio/filckr30k_captions'
TRAINING_SPLIT = path.join(CAPTION_PATH, 'trainimage.txt')
DEV_SPLIT = path.join(CAPTION_PATH, 'devimages.txt')
EVAL_SPLIT = path.join(CAPTION_PATH, 'testimages.txt')
TOKEN_PATH = path.join(CAPTION_PATH, 'captions.token')
OUTPUT_DIR = r'/home/aistudio/PaddlePaddleImageCaption/dict'

def load_spliting(SPLIT_DIR):
    with open(SPLIT_DIR, encoding='utf-8') as f:
        st = f.read().split()
    return st

def load_captions(filter):
    if not isinstance(filter, set):
        filter = set(filter)
    captions = []
    text = []
    with open(TOKEN_PATH, encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '': continue
            filename, context = line.split('\t')
            filename = filename[:-2]
            if filename not in filter: continue
            context = '<start> ' + context[:-1] + ' <stop>'
            captions.append((filename, context))
            text.append(context)
    return captions, text

def generate_training_token(captions, txt):
    filename, texts = zip(*captions)
    filename, texts = list(filename), list(texts)
    # num_word 对 texts_to_sequences 不起作用
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(txt)
    texts = tokenizer.texts_to_sequences(texts)
    w2i = tokenizer.word_index
    i2w = tokenizer.index_word
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'
    
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, padding='post')
    sentense_len = len(texts[0])

    output = []
    st = set()
    for name, text in zip(filename, texts):
        if name not in st:
            output.append((name, text))
            st.add(name)

    return output, w2i, i2w, sentense_len

def gen_eval_set(split, name):

    def word_tok(sentence):
        words = re.split(r'\s|\n|#|,|$|%|&|\.|=|\?|\!|\"|\t', sentence)
        return [w for w in words if w != '']

    captions, _ = load_captions(split)
    captions = [(file_name, word_tok(cap)) for (file_name, cap) in captions]
    filename2cap = {name: [] for name in split}
    for filename, cap in captions:
        filename2cap[filename].append(cap)
    np.save(path.join(OUTPUT_DIR, name), [split, filename2cap])

def main(mx_train=None):
    training_split = load_spliting(TRAINING_SPLIT)
    dev_split = load_spliting(DEV_SPLIT)
    eval_split = load_spliting(EVAL_SPLIT)

    # 生成训练集
    captions, txt = load_captions(training_split)
    captions, word_index, index_word, sentense_len = generate_training_token(captions, txt)
    np.save(path.join(OUTPUT_DIR, 'train_cap.npy'), [captions, word_index, index_word, sentense_len], allow_pickle=True)

    # 生成用于计算BLEU分数的测试集
    gen_eval_set(eval_split, 'eval_data.npy')
    gen_eval_set(dev_split, 'dev_data.npy')

    print('字典大小:{}\n句子最大长度:{}\n<start> idx:{}\n<stop> idx:{}'.format(
        len(word_index), sentense_len, word_index['<start>'], word_index['<stop>']))
    

if __name__ == "__main__":
    main()
# 字典大小:7946
# 句子最大长度:39
# <start> idx:2
# <stop> idx:3

# output:
# 字典大小:18249
# 句子最大长度:80
# <start> idx:2
# <stop> idx:3

# 字典大小:18211
# 句子最大长度:44
# <start> idx:2
# <stop> idx:3