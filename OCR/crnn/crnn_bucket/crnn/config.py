from __future__ import print_function


class Hyperparams(object):
    """
    Hyperparameters for LSTM network
    """
    def __init__(self):
        # Training hyper parameters
        self._train_epoch_size = 30000
        self._eval_epoch_size = 3000
        self._batch_size = 128
        self._num_epoch = 10000
        self._learning_rate = 0.01
        self._momentum = 0.9
        self._num_label = 12
        self._num_hidden = 100
        self._num_lstm_layer = 2
        self._ocr_class = 11
        self._network = 'fmobilenet'
#        self._network = 'simpleConv'
        self._data_path = 'Train_data_bucket'
        self._bucket_len = [20,30]

    @property
    def bucket_len(self):
        return self._bucket_len
    @property
    def train_epoch_size(self):
        return self._train_epoch_size

    @property
    def eval_epoch_size(self):
        return self._eval_epoch_size

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def num_epoch(self):
        return self._num_epoch

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    @property
    def num_label(self):
        return self._num_label

    @property
    def num_hidden(self):
        return self._num_hidden

    @property
    def num_lstm_layer(self):
        return self._num_lstm_layer

    @property
    def data_path(self):
        return self._data_path

#    @property
#    def img_shape(self):
#        return self._img_shape

    @property
    def network(self):
        return self._network

    @property
    def ocr_class(self):
        return self._ocr_class
