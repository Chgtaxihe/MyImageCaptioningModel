"""
    该文件用于绘制训练曲线图
"""

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


def show_extremum_point(values, category='Val', func=max):
    y = func(values)
    x = values.index(y)
    plt.annotate('{}: {} Epoch: {}'.format(category, y, x + 1), xy=(x, y))
    plt.plot(x, y, 'gs')


BLEU_score = extract(context, r'BLEU 分数: ([0-9\.]+)')
meteor_score = extract(context, r'Meteor 分数: ([0-9\.]+)')
diff_sentence = extract(context, r'语句数: ([0-9]+)')
lr = list(map(float, extract(context, r'lr: ([0-9\.e-]+)')))
training_loss = extract(context, r'Epoch loss: ([0-9\.]+)')

plt.figure('TrainData')

plt.subplot(511).set_xticks([])
plt.plot(diff_sentence)
plt.ylabel('diff')

plt.subplot(512).set_xticks([])
plt.plot(lr)
plt.ylabel('lr')

plt.subplot(513).set_xticks([])
plt.plot(training_loss)
plt.ylabel('Loss')
show_extremum_point(training_loss, 'loss', min)

plt.subplot(514).set_xticks([])
plt.plot(BLEU_score)
plt.ylabel('BLEU')
show_extremum_point(BLEU_score, 'BLEU')

plt.subplot(515)
plt.plot(meteor_score)
plt.ylabel('Meteor')
show_extremum_point(meteor_score, 'Meteor')
plt.xlabel('Epoch')


if __name__ == '__main__':
    mx_bleu = max(BLEU_score)
    mx_meteor = max(meteor_score)
    print('BLEU最大值:{} Epoch:{}\nMeteor最大值:{} Epoch:{}'.
          format(mx_bleu, BLEU_score.index(mx_bleu) + 1, mx_meteor, meteor_score.index(mx_meteor) + 1))
    plt.show()
