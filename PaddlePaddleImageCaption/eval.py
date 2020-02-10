from paddle import fluid
from reader import FlickerReader
import numpy as np
import util

USE_CUDNN = True

DICT_DIR = r'/home/aistudio/PaddlePaddleImageCaption/dict'
MODEL_PATH = r'/home/aistudio/save/inference'
BASE_DIR = r'/home/aistudio/flickr30k_feature/'
flickr = FlickerReader(DICT_DIR)
index_word = flickr.index_word
word_index = flickr.word_index

weights = [(1, 0, 0, 0),
           (0, 1, 0, 0),
           (0, 0, 1, 0),
           (0, 0, 0, 1)]

def init_model(eval_model_path):
    place = fluid.CUDAPlace(0) if USE_CUDNN else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [inference_program, feed_target_names, fetch_targets] = \
        fluid.io.load_inference_model(dirname=eval_model_path, executor=exe)
    return exe, inference_program, feed_target_names, fetch_targets

def run_eval(program, exe, eval_data_reader, fetch_list, name='BLEU', weights=(0.25, 0.25, 0.25, 0.25)):
        total_score = 0
        stop_tag = word_index['<stop>']
        for batch_id, batch_data in enumerate(eval_data_reader()):
            data = [img for img, _ in batch_data]
            data = np.stack(data)
            real = [cap for _, cap in batch_data]
            test_result = exe.run(program,
                feed={'input_image': data},
                fetch_list=fetch_list)
            total_score += util.calc_bleu(test_result[0], real, index_word, stop_tag, weights=weights)
        print('{}:{}'.format(name, total_score / (batch_id + 1)))

if __name__ == "__main__":
    e, prog, fee, fet = init_model(MODEL_PATH)
    run_eval(prog, e, flickr.get_test_reader(BASE_DIR), fet, name="BLEU mean")
    for idx, w in enumerate(weights):
        run_eval(prog, e, flickr.get_test_reader(BASE_DIR), fet, name="BLEU@{}".format(idx+1), weights=w)
