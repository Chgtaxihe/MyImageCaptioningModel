import numpy as np
from PIL import Image
from paddle import fluid
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os
import math

DEFAULT_LOG_PATH = r'/home/aistudio/log.txt'
# DEFAULT_LOG_PATH = r'../log.txt'

def translate(index_word, sentence):
    result = []
    for sample in sentence:
        words = []
        for val in sample:
            word = index_word[val]
            if word == '<stop>':
                break
            words.append(word)
            words.append(' ')
        result.append(''.join(words))
    return result

def read_image_norm(paths, shape=(224, 224)):
    images = []
    for path in paths:
        image = Image.open(path)
        image = image.resize(shape,Image.ANTIALIAS) if shape is not None else image
        image = np.array(image).astype(np.float32).transpose([2, 0, 1])
        image = (image / 256.) - 0.5
        images.append(image)
    return np.stack(images)

def calc_bleu(pred, real, index_word, stop_tag, padding_tag=0, weights=(0.25, 0.25, 0.25, 0.25)):
    if isinstance(pred, np.ndarray): pred = pred.tolist()
    smoothie = SmoothingFunction().method4
    sample_cnt = len(pred)
    total_score = 0
    for p, r in zip(pred, real):
        p = [index_word[v] for v in p if all((v != padding_tag, v != stop_tag))]
        # len(p) <= 1 时，bleu无意义
        if len(p) <= 1: continue
        total_score += sentence_bleu(r, p, smoothing_function=smoothie, weights=weights)
    return total_score / sample_cnt

def log(context, end='\n', log_dir=DEFAULT_LOG_PATH):
    print(context, end=end)
    with open(log_dir, 'a') as f:
        f.write(context + end)
