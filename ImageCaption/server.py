from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from paddle import fluid

DIC_PATH = '/paddle/word_dict.npy'
MODEL_PATH = '/paddle/infer_bleu'
INDEX = '/paddle/index.html'

DIC_PATH = 'F:/Dataset/ai_challenge_dict/word_dict.npy'
MODEL_PATH = 'F:/infer_bleu'
INDEX = 'C:/Users/cgxih/Desktop/index.html'

index_word = []

exe, eval_prog, feed_target_names, fetch_targets = [None] * 4
app = Flask(__name__)

def process_image(img):
    img = img.resize([224, 224], Image.ANTIALIAS)
    img = np.array(img, dtype='float32')
    if len(img.shape) != 3:
        return None
    img = img.transpose((2, 0, 1)) / 255
    return img

def convert(p):
    result = []
    for idx in p:
        if idx == 3: break
        if idx == 0: continue
        result.append(index_word[idx])
    return ' '.join(result)

def init():
    global index_word, exe, eval_prog, feed_target_names, fetch_targets
    _, index_word = np.load(DIC_PATH, allow_pickle=True)
    exe = fluid.Executor(fluid.CPUPlace())
    [eval_prog, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
        dirname=MODEL_PATH, executor=exe)

def pred(fp):
    img = process_image(Image.open(fp))
    img = np.expand_dims(img, 0)
    result = exe.run(eval_prog, feed={feed_target_names[0]: img}, fetch_list=fetch_targets)[0]
    print(result[0].tolist())
    result = convert(result[0].tolist())
    return result

@app.route('/infer', methods=['POST'])
def infer():
    fp = request.files['img']
    return pred(fp)

@app.route('/')
def main():
    with open(INDEX, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    init()
    app.run(host='0.0.0.0', port=8987, debug=True)
