import os.path as path
import re

ROOT_PATH = r'F:\Dataset\flickr30k'
TOKEN_PATH = path.join(ROOT_PATH, 'captions.token.src')
EXPORT_PATH = path.join(ROOT_PATH, 'captions.token')
MAX_LENGTH = 30

symbol = re.compile(r"[^a-zA-Z]+")


def word_tok(text):
    text = re.sub(symbol, " ", text.strip())
    words = text.split()
    return words


lines = []
removed_line = []
out = open(EXPORT_PATH, 'w', encoding='utf-8')
with open(TOKEN_PATH, 'r', encoding='utf-8') as f:
    for line in f:   
        if len(line) <= 1:
            continue
        name, content = line.split('\t')
        if len(word_tok(content)) <= MAX_LENGTH:
            out.write(line)
        else:
            removed_line.append(name)
out.close()
print(removed_line[:5])
print('删除了{}句话'.format(len(removed_line)))

# 删除了7004句话
if __name__ == '__main__':
    pass
