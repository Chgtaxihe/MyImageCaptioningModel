import numpy as np
import os.path as path
import re

CAPTION_PATH = r'F:\Dataset\filckr30k'
TOKEN_PATH = path.join(CAPTION_PATH, 'captions.token.src')
EXPORT_PATH = path.join(CAPTION_PATH, 'captions.token')
MAX_LENGTH = 40

def word_tok(sentence):
    words = re.split(r'\s|\n|#|,|$|%|&|\.|=|\?|\!|\"|\t', sentence)
    return [w for w in words if w != '']

lines = []
removed_line = []
out = open(EXPORT_PATH, 'w', encoding='utf-8')
with open(TOKEN_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        name, content = line.split('\t')
        if len(word_tok(content)) <= MAX_LENGTH:
            out.write(line)
        else:
            removed_line.append(name)
out.close()
print(removed_line)
