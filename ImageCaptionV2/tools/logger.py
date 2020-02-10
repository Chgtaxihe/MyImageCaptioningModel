import json
import os

import config
from tools import util


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton


path = config.log['log_path']


@singleton
class Logger:

    def _first_init(self):
        self.is_first_init = True
        self._conf = {'epoch': 1, 'best_bleu': 0, 'best_meteor': 0,
                      'train_encoder': config.model['encoder']['encoder_trainable']}
        self._save_conf()
        
    def _load_config(self):
        dic = util.read_file(os.path.join(path, 'config'))
        dic = json.loads(dic)
        return dic

    def __init__(self):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, 'config')):
            self._first_init()
        else:
            self._conf = self._load_config()
        self.is_first_init = (self.epoch == 1)

    def _save_conf(self):
        util.write_file(os.path.join(path, 'config'), json.dumps(self._conf))
    
    @property
    def epoch(self):
        return self._conf['epoch']

    @epoch.setter
    def epoch(self, val):
        self._conf['epoch'] = val
        self._save_conf()

    @property
    def best_bleu(self):
        return self._conf['best_bleu']

    @best_bleu.setter
    def best_bleu(self, val):
        self._conf['best_bleu'] = val
        self._save_conf()

    @property
    def best_meteor(self):
        return self._conf['best_meteor']

    @best_meteor.setter
    def best_meteor(self, val):
        self._conf['best_meteor'] = val
        self._save_conf()

    @property
    def train_encoder(self):
        return self._conf.get('train_encoder', False)

    @train_encoder.setter
    def train_encoder(self, val):
        self._conf['train_encoder'] = val
        self._save_conf()

    @staticmethod
    def log(content, end='\n'):
        logfile = os.path.join(path, 'log.txt')
        print(content, end=end)
        with open(logfile, 'a') as f:
            f.write(content + end)
