from __future__ import print_function

import numpy as np
import mxnet as mx
import pdb
import os
import cv2
import random
from config import Hyperparams

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names=list(), label=list()):
        self._data = data
        self._label = label
        self._data_names = data_names
        self._label_names = label_names
        self.pad = 0

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def data_names(self):
        return self._data_names

    @property
    def label_names(self):
        return self._label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self._data_names, self._data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self._label_names, self._label)]


def img_pre(img,channel,height,width):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img=cv2.resize(img,(width,height))
    img=img.reshape(channel,height,width)
    return img

def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret

class OCRIter(mx.io.DataIter):
    """
    Iterator class for generating captcha image data
    """
    def __init__(self, count, batch_size, lstm_init_states,data_path, name):
        """
        Parameters
        ----------
        count: int
            Number of batches to produce for one epoch
        batch_size: int
        lstm_init_states: list of tuple(str, tuple)
            A list of tuples with [0] name and [1] shape of each LSTM init state
        captcha MPCaptcha
            Captcha image generator. Can be MPCaptcha or any other class providing .shape and .get() interface
        name: str
        """
        super(OCRIter, self).__init__()
        hp = Hyperparams()
        self.batch_size = batch_size
        self.count = count
        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]
        self.name = name
        self.lists=os.listdir(data_path)
        self.data_path=data_path
        self.point=0
        self.channel,self.height,self.width = hp.img_shape
        self.num_hidden = hp.num_hidden
        self.num_label =hp.num_label
    #def __iter__(self):
    #    return self


    def get_data(self,):
        imgs=[]
        label_list=[]
        labels=np.zeros(self.num_label)
        lists=self.lists[self.point:self.point+self.batch_size]
        self.point += self.batch_size
        
        for i,files in enumerate(lists):
            img = cv2.imread(os.path.join(self.data_path,files))
            img = img_pre(img,self.channel,self.height,self.width)
            imgs.append(img)
            label_list.append( np.array((int(files[0])+1,int(files[1])+1,int(files[2])+1,int(files[3])+1)) ) 
        return imgs,label_list
    def reset(self,):
         self.point=0
         random.shuffle(self.lists)

    @property
    def provide_data(self):
        return [('data', (self.batch_size,self.channel, self.height,self.width)),('l0_init_c', (self.batch_size, self.num_hidden)),
                ('l1_init_c', (self.batch_size, self.num_hidden)),('l0_init_h', (self.batch_size, self.num_hidden)),('l1_init_h', (self.batch_size, self.num_hidden))]

    @property
    def provide_label(self):
        return [('label', (self.batch_size, self.num_label))]


    def next(self,):
        init_state_names = [x[0] for x in self.init_states]
        if self.point>len(self.lists)-self.batch_size:
            self.reset()
            raise StopIteration
        data = []
        label = []
        data,label = self.get_data()
        data_all = [mx.nd.array(data)] + self.init_state_arrays
        label_all = [mx.nd.array(label)]
        data_names = ['data'] + init_state_names
        label_names = ['label']
        return mx.io.DataBatch( data_all, label_all )
