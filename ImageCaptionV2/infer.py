import numpy as np
import os
import requests
import sys
from io import BytesIO
from paddle import fluid
from PIL import Image

import config
import evaluate
import reader
import train
from tools import util
from model.MobileNetV2 import MobileNetV2


exe = fluid.Executor(fluid.CUDAPlace(0))

model_name = 'show_and_tell'
info = {
    'show_and_tell': {
        'use_pooling': True,
        'squeeze': True
    },
    'base': {
        'use_pooling': False
    }
}

def download_image(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ConnectionError('下载图片 ' + url + ' 失败 %d' % resp.status_code)
    img = Image.open(BytesIO(resp.content))
    img = reader.process_image(img)
    return img


def extract_feature(img):
    cfg = info[model_name]
    startup, extra_prog = fluid.Program(), fluid.Program()
    with fluid.program_guard(extra_prog, startup):
        with fluid.unique_name.guard():
            model = MobileNetV2(use_pooling=cfg.get('use_pooling'))
            x = fluid.data('feed_in', [1, 3, 224, 224])
            feature = model.net(x)
            if cfg.get('squeeze', False):
                feature = fluid.layers.squeeze(feature, axes=[2, 3])
    exe.run(startup)
    fluid.io.load_vars(exe, config.dc['PretrainedMobileNetPath'], extra_prog,
                       predicate=util.get_predicate(config.dc['PretrainedMobileNetPath']))
    feature = exe.run(extra_prog, feed={'feed_in': img}, fetch_list=[feature])[0]
    return feature


def main(url):
    img = download_image(url)
    img = np.expand_dims(img, 0)
    if not config.train['use_raw_image']:
        img = extract_feature(img)
    caption, startup_prog, eval_prog = train.eval_net(config.train['use_raw_image'])

    exe.run(startup_prog)
    p = os.path.join(config.train['checkpoint_path'], 'checkpoint_best_bleu')
    fluid.io.load_vars(exe, p, eval_prog, predicate=util.get_predicate(p))

    result = exe.run(eval_prog, feed={'image': img}, fetch_list=[caption])[0][0]
    result = evaluate.filter(result)
    sentence = evaluate.words2sentence(result)
    print(sentence)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
