import paddle.fluid as fluid
from model import BaseModel
from reader import FlickerReader
from PaddleTools.ImplNetwork import ImplNetwork
from PaddleTools.trainer import Trainer
import numpy as np
import util
import time
import os

DICT_DIR = r'/home/aistudio/PaddlePaddleImageCaption/dict'
BASE_DIR = r'/home/aistudio/flickr30k_feature'
WORKPLACE = r'/home/aistudio/save'
# PRETRAIN_MODEL_DIR = r'/home/aistudio/MobileNetV2_pretrained'
PRETRAIN_MODEL_DIR = None
test_image_path = os.path.join(BASE_DIR, os.listdir(BASE_DIR)[0])
BATCH_SIZE = 64
LOG_PER_N_STEP = 50 #这样子日志会干净一点
training_epoch = 200
USE_CUDNN = True
EXTRACT_VARS = True # 同时调用save_vars

GRADIENT_CLIP_NORM = 0.0001

# 返回Monmentum优化器
def monmentum():
    LEARNING_RATE = 0.0001
    MONMENTUM = 0.9
    return fluid.optimizer.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MONMENTUM)

# 返回Adam优化器
def adam():
    LEARNING_RATE = 0.0001
    return fluid.optimizer.Adam(learning_rate=LEARNING_RATE)

flickr = FlickerReader(DICT_DIR)
word_index = flickr.word_index
index_word = flickr.index_word
sentence_len = flickr.sentence_len
vocab_size = len(word_index)

class TrainingNetwork(ImplNetwork):

    def build(self):
        # images = fluid.layers.data('input_image', shape=[None, 3, 224, 224])
        images = fluid.layers.data('input_image', shape=[None, 1280, 7, 7])
        model = BaseModel.ImageCaptionModle(use_cudnn=USE_CUDNN)
        captions = fluid.layers.data('input_captions', shape=[None, sentence_len], dtype='int64')
        loss = model.build_network(images, captions)
        # fluid.clip.set_gradient_clip(
        #     fluid.clip.GradientClipByNorm(GRADIENT_CLIP_NORM))
        optimizer = adam()
        optimizer.minimize(loss)
        self.loss = loss
        self.input = [images, captions]

    def get_input(self):
        return self.input
    
    def get_loss(self):
        return self.loss

class EvalNetwork(ImplNetwork):

    def build(self):
        # images = fluid.layers.data('input_image', shape=[None, 3, 224, 224])
        images = fluid.layers.data('input_image', shape=[None, 1280, 7, 7])
        model = BaseModel.ImageCaptionModle(use_cudnn=USE_CUDNN)
        result = model.build_sampler(images)
        self.output = [result, ]
        self.input = [images, ]
    
    def get_output(self):
        return self.output
    
    def get_input(self):
        return self.input


def train():
    
    util.log("用于测试的图片:%s"%test_image_path)
    
    training_network = TrainingNetwork()
    eval_network = EvalNetwork()

    def eval_func(exe, eval_prog):
        util.log('DevSet上的得分:', end=' ')
        run_eval(eval_prog, exe, flickr.get_dev_reader(base_dir=BASE_DIR, batch_size=BATCH_SIZE))
        
        test_iamge = np.load(test_image_path, allow_pickle=True)
        test_result = exe.run(program=eval_prog, feed={'input_image':test_iamge}, fetch_list=eval_network.output)
        util.log('测试输出: {}'.format(util.translate(index_word, test_result[0])))

    def run_eval(program, exe, eval_data_reader):
        start_time = time.time()
        total_score = 0
        stop_tag = word_index['<stop>']
        for batch_id, batch_data in enumerate(eval_data_reader()):
            data = [img for img, _ in batch_data]
            data = np.stack(data)
            real = [cap for _, cap in batch_data]
            test_result = exe.run(program,
                feed={'input_image': data},
                fetch_list=eval_network.output)
            total_score += util.calc_bleu(test_result[0], real, index_word, stop_tag)
            
        util.log('BLEU 分数: {:.7f} 耗时: {:.2f}秒'.format(total_score / (batch_id + 1), time.time()-start_time))

    trainer = Trainer(WORKPLACE, training_network, flickr.get_training_reader(BASE_DIR), use_cudnn=USE_CUDNN,
        batch_size=BATCH_SIZE, eval_network=eval_network, auto_restore=True,
        pretrain_dir=PRETRAIN_MODEL_DIR, logger=util.log, max_epoch=training_epoch, extract_vars=EXTRACT_VARS)
    
    trainer.init_network()
    # trainer.restore_model_by_param(os.path.join(WORKPLACE, 'saved_vars'))
    print("ok")
    trainer.train(eval_func=eval_func, log_per_n_step=LOG_PER_N_STEP)
    # trainer.save_infer_model()

def send_alert(content, subject='信息通知'):

    return None


if __name__ == "__main__":
    try:
        # send_alert("训练开始")
        train()
    except Exception as e:
        send_alert("Aistudio训练异常\n" + str(e))
        print(e)
        exit(0)
    sendemail("正常结束")
        
        
