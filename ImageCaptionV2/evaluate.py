import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from paddle import fluid

import config
import train
from tools import util
from reader import DataReader

index_word = DataReader().index_word
stop_tag = config.data['stop_idx']
padding_tag = config.data['padding_idx']


def filter(p):
    """
        把索引list转换为单词list
    """
    result = []
    for idx in p:
        if idx == stop_tag:
            break
        if idx == padding_tag: continue
        result.append(index_word[idx])
    return result


def calc_bleu(pred, real, weights=(0.25, 0.25, 0.25, 0.25)):
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    total_score = 0
    for p, r in zip(pred, real):
        p = filter(p)
        if len(p) <= 1:
            continue
        total_score += sentence_bleu(r, p, smoothing_function=SmoothingFunction().method4, weights=weights)
    return total_score / len(pred)


def words2sentence(words):
    return ' '.join(words)


def calc_meteor(pred, real):
    if isinstance(pred, np.ndarray): pred = pred.tolist()
    total_score = 0
    for p, r in zip(pred, real):
        p = words2sentence(filter(p))
        r = [words2sentence(e) for e in r]
        total_score += meteor_score(r, p)
    return total_score / len(pred)


def evaluate():
    places = fluid.CUDAPlace(0)
    exe = fluid.Executor(places)

    [eval_prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                    dirname=os.path.join(config.train['checkpoint_path'], 'infer_bleu'),
                                                    executor=exe)

    batch_size = config.train['batch_size']
    dr = DataReader()
    dr = dr.get_reader(batch_size, 'test')
    bleu_score, mscore = [0] * 5, 0
    bleu_vec = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
    use_meteor = config.train['calc_meteor']
    sentence_said = set()
    for l, data in enumerate(dr()):
        img, real_cap = zip(*data)
        cp = exe.run(eval_prog, feed={feed_target_names[0]: img}, fetch_list=fetch_targets)[0]
        for idx, vec in enumerate(bleu_vec):
            bleu_score[idx] += calc_bleu(cp, real_cap, vec)
        if use_meteor:
            mscore += calc_meteor(cp, real_cap)
        if config.evaluate['sentence_statistics']:
            for p in cp:
                p = words2sentence(filter(p))
                sentence_said.add(p)
    for i in range(len(bleu_score)):
        bleu_score[i] /= l + 1
    bleu_score[4] = sum(bleu_score[:-1]) / 4
    mscore /= l + 1
    print('Test set:\nMeteor 分数: {:.7f}'.format(mscore))
    print('BLEU [{:.7f}, {:.7f}, {:.7f}, {:.7f}] {:.7f}'.format(*bleu_score))
    if config.evaluate['sentence_statistics']:
        print('模型一共说了{}句不同的话'.format(len(sentence_said)))


if __name__ == '__main__':
    evaluate()
