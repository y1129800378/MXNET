from __future__ import print_function
import cv2
import sys
import pdb
import argparse
import mxnet as mx
import numpy as np
from crnn.fit import lstm
from collections import namedtuple
from crnn.OcrIter import SimpleBatch
from crnn.config import Hyperparams
from crnn.fit.ctc_metrics import CtcMetrics

class OCR_recognition(object):
    def __init__(self,model_name):
        hp = Hyperparams()
        self.num_hidden = hp.num_hidden
        self.num_lstm_layer = hp.num_lstm_layer
        self.channel,self.height,self.width = hp.img_shape
        self.mod = self.load_model(model_name)
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_states]

    def load_model(self,model_name,batch_size=1):
        name,epoch = model_name.split(',')
        sym, arg_params, aux_params = mx.model.load_checkpoint(name, int(epoch))
        # We don't need CTC loss for prediction, just a simple softmax will suffice.
        # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
        pred_fc = sym.get_internals()['pred_fc_output']
        sym = mx.sym.softmax(data=pred_fc)
        data_names = ['data']
        data_names += ['l%d_init_c' % l for l in range(self.num_lstm_layer)]
        data_names += ['l%d_init_h' % l for l in range(self.num_lstm_layer)]
        init_c = [('l%d_init_c' % l, (batch_size, self.num_hidden)) for l in range(self.num_lstm_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, self.num_hidden)) for l in range(self.num_lstm_layer)]
        self.init_states = init_c + init_h
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (batch_size,self.channel,self.height,self.width))] + self.init_states)
        mod.set_params(arg_params, aux_params, allow_missing=False)
        return mod

    def _preprocess_image(self,img_path):
        img = cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img=cv2.resize(img,(self.width,self.height))
        img=img.reshape((1,self.channel,self.height,self.width))
        img = [mx.nd.array(img)] + self.init_state_arrays
        return img

    def predict(self,img_path):
        '''
        use mx.io.NDArrayIter and mod.predict to predict
        '''
        img = self._preprocess_image(img_path)
        img = mx.io.NDArrayIter(data=img,label=None,batch_size=1)
        res = self.mod.predict(eval_data=img, num_batch=1)
        res=res.asnumpy()
        prediction = CtcMetrics.ctc_label(np.argmax(res, axis=-1).tolist())
        prediction = [p-1 for p in prediction]
        return prediction

    def predict_DataIter(self,img_path):
        '''
        super mx.io.DataIter and mod.forward to predict
        '''
        img = self._preprocess_image(img_path)
        img = Io_class(img)
        self.mod.forward(img)
        res = self.mod.get_outputs()[0].asnumpy()
        prediction = CtcMetrics.ctc_label(np.argmax(res, axis=-1).tolist())
        prediction = [p-1 for p in prediction]
        return prediction

class Io_class(mx.io.DataIter):
    def __init__(self, _datas):
        super(Io_class, self).__init__()
        self.datas = _datas

    @property
    def data(self):
        return self.datas


if __name__ == '__main__':
    model_name = sys.argv[1]
    img_path =sys.argv[2]
    ocr = OCR_recognition(model_name)
    print(ocr.predict(img_path))
    print(ocr.predict_DataIter(img_path))
