from paddle import fluid
import os
from os import path
import requests
import sys
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from paddle.fluid.framework import Parameter

DICT_DIR = r'/home/aistudio/PaddlePaddleImageCaption/dict'
MODEL_PATH = r'/home/aistudio/save/inference'
MOBILE_NET_PRETRAIN = r'/home/aistudio/MobileNetV2_pretrained'
USE_CUDNN = True

place = fluid.CUDAPlace(0) if USE_CUDNN else fluid.CPUPlace()
exe = fluid.Executor(place)

train_cap_path = path.join(DICT_DIR, 'train_cap.npy')
captions, word_index, index_word, sentence_len = np.load(train_cap_path, allow_pickle=True)

def download_image(urls):
    images = []
    for url in urls:
        resp = requests.get(url)
        if resp.status_code != 200:
            print('下载图片 ' + url + ' 失败 %d' % resp.status_code)
            continue
        img = Image.open(BytesIO(resp.content))
        plt.figure()
        plt.imshow(img)
        img = image_norm(img)
        images.append(img)
    return np.stack(images)

def _load_pretrain_param(executor, dirname, main_program):
    def predicate(var):
        if not isinstance(var, Parameter): return False
        file_path = os.path.normpath(os.path.join(dirname, var.name))
        if not os.path.isfile(file_path):
            return False
        return True

    fluid.io.load_vars(executor, dirname, main_program=main_program,
        predicate=predicate)

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
_img_mean = np.array(_mean).reshape((3, 1, 1))
_img_std = np.array(_std).reshape((3, 1, 1))

def image_norm(image, shape=[224, 224]):
    image = image.resize(shape,Image.ANTIALIAS) if shape is not None else image
    image = np.array(image).astype(np.float32).transpose([2, 0, 1])
    image = (image / 255.) - _img_mean
    image /= _img_std
    return image.astype(np.float32)

def init_mobilenet():
    from model.MobileNetV2 import MobileNetV2
    prog = fluid.Program()
    startup = fluid.Program()
    model = MobileNetV2()
    with fluid.program_guard(prog, startup):
        data = fluid.layers.data('image', [3, 224, 224])
        feature = model.net(data)
    exe.run(startup)
    _load_pretrain_param(exe, MOBILE_NET_PRETRAIN, prog)
    return prog, feature

def init_infer_model(eval_model_path):
    [inference_program, feed_target_names, fetch_targets] = \
        fluid.io.load_inference_model(dirname=eval_model_path, executor=exe)
    return inference_program, feed_target_names, fetch_targets

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
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法 python infer.py URL")
    img = download_image([sys.argv[1]])
    infer_prog, feed, fetch = init_infer_model(MODEL_PATH)
    mobilenet_prog, f = init_mobilenet()

    feature = exe.run(mobilenet_prog, feed={'image': img}, fetch_list=[f])[0]
    result = exe.run(infer_prog, feed={feed[0]:feature}, fetch_list=fetch)[0]
    print(translate(index_word, result))
