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


exe = fluid.Executor(fluid.CUDAPlace(0))


def download_image(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ConnectionError('下载图片 ' + url + ' 失败 %d' % resp.status_code)
    img = Image.open(BytesIO(resp.content))
    img = reader.process_image(img)
    return img


def main(url):
    img = download_image(url)
    img = np.expand_dims(img, 0)
    [eval_prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
        dirname=os.path.join(config.train['checkpoint_path'], 'infer'),
        executor=exe)
    result = exe.run(eval_prog, feed={feed_target_names[0]: img}, fetch_list=fetch_targets)[0]
    print(result[0].tolist())
    result = evaluate.filter(result[0].tolist())
    sentence = evaluate.words2sentence(result)
    print(sentence)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    main(sys.argv[1])
