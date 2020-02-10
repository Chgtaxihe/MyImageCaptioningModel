import json
import numpy as np
import os
import time
from paddle import fluid
from paddle.fluid.framework import Parameter
from PaddleTools import Utils


class Trainer():

    DEFAULT_CONFIG = {'CurrentEpoch': 1}

    def __init__(self, workplace, train_network, training_data_reader,  
        eval_network=None, max_epoch=200, batch_size=32, use_cudnn=True, pretrain_dir=None,
        logger=Utils.print_logger, extract_vars=False, auto_restore=True):

        self.workplace = os.path.expanduser(workplace)
        self.train_network = train_network
        self.training_data_reader = training_data_reader
        self.eval_network = eval_network
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.use_cudnn = use_cudnn
        self.pretrain_dir = pretrain_dir
        self.extract_vars = extract_vars
        self.auto_restore = auto_restore
        self.log = logger

        self.save_dir = os.path.join(self.workplace, 'checkpoint')
        self.save_var_dir = os.path.join(self.workplace, 'saved_vars')
        self.inference_dir = os.path.join(self.workplace, 'inference')
        self.config = Trainer.DEFAULT_CONFIG

        if not os.path.exists(workplace): os.makedirs(workplace)

    # 使用orthogonal初始化Lstm的权重
    # 然而PaddlePaddle并没有相关的API~
    def _init_lstm(self):
        w = [Utils.orthogonal([2816, 1024]) for _ in range(4)]
        w = np.concatenate(w, axis=1)
        pd_var = fluid.global_scope().find_var('lstm_loop_weight')
        pd_param = pd_var.get_tensor()
        pd_param.set(w, self.places)

    def init_network(self):
        self.train_prog = fluid.Program()
        train_startup = fluid.Program()

        # PaddlePaddle Fluid中使用 fluid.unique_name 包来随机初始化用户未定义的参数名称。
        # 通过 fluid.unique_name.guard 可以确保多次调用某函数参数初始化的名称一致。
        with fluid.program_guard(self.train_prog, train_startup):
            with fluid.unique_name.guard():
                self.train_network.build()
                self.train_feed_in = self.train_network.get_input()
                self.train_dataloader = fluid.io.DataLoader.from_generator(feed_list=self.train_feed_in, capacity=64)
                self.train_loss = self.train_network.get_loss()

        if self.eval_network is not None:
            self.eval_prog = fluid.Program()
            eval_startup = fluid.Program()
            with fluid.program_guard(self.eval_prog, eval_startup):
                with fluid.unique_name.guard():
                    self.eval_network.build()
                    self.eval_feed_in = self.eval_network.get_input()
                    self.eval_output = self.eval_network.get_output()
        
        self.places = fluid.CUDAPlace(0) if self.use_cudnn else fluid.CPUPlace()
        self.executor = fluid.Executor(self.places)
        
        self.executor.run(train_startup)
        if self.eval_network is not None: self.executor.run(eval_startup)
        
        self._load_conf()
        if self.config.get('CurrentEpoch', 1) == 1:
            self._init_lstm() # lstm使用正交初始化 
            if self.pretrain_dir is not None: # 加载预训练模型
                Trainer._load_pretrain_param(self.executor, self.pretrain_dir, main_program=self.train_prog)

        elif self.auto_restore:
            self._restore_model(self.executor, self.train_prog) # 恢复模型

    def _restore_model(self, executor, program):
        # 不需要恢复
        if not os.path.isdir(self.save_dir):
            return
        fluid.io.load_persistables(executor, self.save_dir, main_program=program)
    
    def restore_model_by_param(self, dir):
        self._load_pretrain_param(self.executor, dir, self.train_prog)

    def _load_conf(self):
        conf_path = os.path.join(self.workplace, 'config.json')
        # 使用默认配置
        if not os.path.isfile(conf_path):
            return
        context = Utils.read_file(conf_path)
        self.config = json.loads(context)

    def _update_conf(self, **kw):
        self.config.update(kw)
        conf_path = os.path.join(self.workplace, 'config.json')
        Utils.write_file(conf_path, json.dumps(self.config))

    def save_infer_model(self):
        feed_in_var_name = [var.name for var in self.eval_feed_in]
        fluid.io.save_inference_model(self.inference_dir, feed_in_var_name, 
            self.eval_output, self.executor, main_program=self.eval_prog)

    def train(self, log_per_n_step=50, eval_func=None):
        for epoch in range(self.config['CurrentEpoch'], self.max_epoch+1):
            self.log('开始运行Epoch {}'.format(epoch))
            
            epoch_loss = 0
            start_time = time.time()
            
            self.train_dataloader.set_sample_generator(self.training_data_reader, self.batch_size, drop_last=True, places=self.places)
            
            for batch, data in enumerate(self.train_dataloader()):
                batch_loss = self.executor.run(program=self.train_prog, feed=data, 
                    fetch_list=[self.train_loss])
                
                if self._has_nan(batch_loss):
                    self.log(' '*4 + '在 Batch {} 出现 Nan,训练终止'.format(batch))
                    raise ValueError('Epoch {} 在 Batch {} 出现 Nan,训练终止'.format(epoch, batch))
                epoch_loss += batch_loss[0][0]

                if (batch + 1) % log_per_n_step == 0:
                    self.log(' '*4 + 'Batch {} MeanLoss {:.6f} Batch Loss {:.4f}'.format(
                        batch+1, epoch_loss/(batch+1)/self.batch_size, batch_loss[0][0]/self.batch_size))
                    
            
            self.log('Epoch loss:{:.6f}'.format(epoch_loss/(batch+1)/self.batch_size))

            fluid.io.save_persistables(self.executor, self.save_dir, main_program=self.train_prog)
            
            if epoch % 5 == 0 and epoch != 0:
                fluid.io.save_persistables(self.executor, self.save_dir + "%d"%epoch, main_program=self.train_prog)
                
            if self.extract_vars:
                fluid.io.save_params(self.executor, self.save_var_dir, main_program=self.train_prog)

            if self.eval_network is not None: 
                self.save_infer_model()

            # TODO
            # eval 待修改
            if eval_func is not None:
                eval_func(self.executor, self.eval_prog)

            self.log('Epoch耗时:{:.2f}秒'.format(time.time() - start_time))
            self._update_conf(CurrentEpoch=epoch + 1)
        
    def _has_nan(self, val):      
        return np.isnan(val).any()

    @staticmethod
    def _load_pretrain_param(executor, dirname, main_program):
        def predicate(var):
            if not isinstance(var, Parameter): return False
            file_path = os.path.normpath(os.path.join(dirname, var.name))
            if not os.path.isfile(file_path):
                return False
            return True

        fluid.io.load_vars(executor, dirname, main_program=main_program,
            predicate=predicate)

