# %matplotlib inline
import re
from matplotlib import pyplot as plt

LOG_PATH = r'C:\Users\cgxih\Desktop\log.txt'

def read_file(p):
    with open(p, encoding='utf-8') as f:
        context = f.read()
    return context

context = read_file(LOG_PATH)

def extract(txt, pattern):
    match = re.findall(pattern, txt)
    return list(map(float, match))

BLEU_score = extract(context, r'BLEU 分数: ([0-9\.]+)')
meteor_score = extract(context, r'Meteor 分数: ([0-9\.]+)')
diff_sentence = extract(context, r'语句数: ([0-9]+)')
lr = extract(context, r'lr: ([0-9\.e-]+)')
training_loss = extract(context, r'Epoch loss: ([0-9\.]+)')

plt.figure('TrainData')
plt.subplot(511)
plt.plot(BLEU_score)
plt.ylabel('BLEU')
# plt.xlabel('Epoch')

plt.subplot(512)
plt.plot(meteor_score)
plt.ylabel('Meteor')
# plt.xlabel('Epoch')

plt.subplot(513)
plt.plot(diff_sentence)
plt.ylabel('diff')

plt.subplot(514)
plt.plot(lr)
plt.ylabel('lr')

plt.subplot(515)
plt.plot(training_loss)
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.show()
