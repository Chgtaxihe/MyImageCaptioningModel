import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from paddle import fluid

import config
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
        if pred.dtype == 'float32':
            pred = np.rint(pred).astype('int32')
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

def evaluate():
    places = fluid.CUDAPlace(0)
    exe = fluid.Executor(places)

    [eval_prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
                                                    dirname=os.path.join(config.train['checkpoint_path'], 'infer_meteor'),
                                                    executor=exe)
    exe = fluid.ParallelExecutor(use_cuda=True,
                                 main_program=eval_prog)
    batch_size = config.train['batch_size']
    dr = DataReader()
    dr = dr.get_reader(batch_size, 'test')
    bleu_score = [0] * 5
    bleu_vec = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
    sentence_said = set()
    for l, data in enumerate(dr()):
        img, real_cap = zip(*data)
        cp = exe.run(feed={feed_target_names[0]: np.array(img, dtype='float32')}, fetch_list=fetch_targets)[0]
        for idx, vec in enumerate(bleu_vec):
            bleu_score[idx] += calc_bleu(cp, real_cap, vec)
        if config.evaluate['sentence_statistics']:
            for p in cp:
                p = words2sentence(filter(p))
                sentence_said.add(p)
    for i in range(len(bleu_score)):
        bleu_score[i] /= l + 1
    bleu_score[4] = sum(bleu_score[:-1]) / 4
    print('BLEU [{:.7f}, {:.7f}, {:.7f}, {:.7f}] {:.7f}'.format(*bleu_score))
    if config.evaluate['sentence_statistics']:
        print('模型一共说了{}句不同的话'.format(len(sentence_said)))


if __name__ == '__main__':
    evaluate()
