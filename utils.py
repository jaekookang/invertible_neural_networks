'''Utilities

2020-11-17 first created
'''

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from time import time, strftime, gmtime
import tensorflow as tf
tfk = tf.keras
tfkc = tfk.callbacks


class NBatchLogger(tfkc.Callback):
    '''A Logger that logs the average performance per `display` steps.
    See: https://gist.github.com/jaekookang/7e2ca4dc2b1ab10dbb80b9e65ca91179
    '''

    def __init__(self, n_display, max_epoch, save_dir=None, suffix=None, silent=False):
        self.epoch = 0
        self.display = n_display
        self.max_epoch = max_epoch
        self.logs = {}
        self.save_dir = save_dir
        self.silent = silent
        if self.save_dir is not None:
            assert os.path.exists(self.save_dir), Exception(
                f'Path:{self.save_dir} does not exist!')
            fname = 'train.log'
            if suffix is not None:
                fname = f'train_{suffix}.log'
            self.fid = open(os.path.join(save_dir, fname), 'w')
        self.t0 = time()

    def on_train_begin(self, logs={}):
        logs = logs or self.logs
        txt = f'=== Started at {self.get_time()} ==='
        self.write_log(txt)
        if not self.silent:
            print(txt)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if (self.epoch % self.display == 0) | (self.epoch == 1):
            txt = f' {self.get_time()} | Epoch: {self.epoch}/{self.max_epoch} | '
            if not self.silent:
                print(txt, end='')

            for i, key in enumerate(logs.keys()):
                if (i+1) == len(logs.keys()):
                    _txt = f'{key}={logs[key]:4f}'
                    if not self.silent:
                        print(_txt, end='\n')
                else:
                    _txt = f'{key}={logs[key]:4f} '
                    if not self.silent:
                        print(_txt, end='')
                txt = txt + _txt
            self.write_log(txt)
        self.logs = logs

    def on_train_end(self, logs={}):
        logs = logs or self.logs
        t1 = time()
        txt = f'=== Time elapsed: {(t1-self.t0)/60:.4f} min (loss:{logs["loss"]:4f}) ==='
        if not self.silent:
            print(txt)
        self.write_log(txt)

    def get_time(self):
        return strftime('%Y-%m-%d %Hh:%Mm:%Ss', gmtime())

    def write_log(self, txt):
        if self.save_dir is not None:
            self.fid.write(txt+'\n')
            self.fid.flush()