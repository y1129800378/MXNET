from __future__ import print_function

import argparse
import logging
import os

from crnn.config import Hyperparams
from crnn.fit.ctc_metrics import CtcMetrics
import mxnet as mx

import crnn.fit.lstm as lstm
from crnn.OcrIter import OCRIter


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", help="'ctc' or 'warpctc' loss [Default 'ctc']", default='ctc')
    parser.add_argument("--cpu",
                        help="Number of CPUs for training [Default 8]. Ignored if --gpu is specified.",
                        type=int, default=8)
    parser.add_argument("--gpu", help="Number of GPUs for training [Default 0]", type=int)
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'mobilenet']", default='checkpoint/mobilenet')
    parser.add_argument("--resume", help="Checkpoint resume [Default 'checkpoint/mobilenet,10']", default=None)
    parser.add_argument("--save_epoch", help="Number of epoch to save model [Default 10]", default=10,type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not any(args.loss == s for s in ['ctc', 'warpctc']):
        raise ValueError("Invalid loss '{}' (must be 'ctc' or 'warpctc')".format(args.loss))
    hp = Hyperparams()

    try:
        if args.resume:
            model_path,epoch=args.resume.split(",")
            _,arg_params, aux_params = mx.model.load_checkpoint(model_path,int(epoch))
        else:
            arg_params, aux_params = None,None

        if args.gpu:
            contexts = [mx.context.gpu(i) for i in range(args.gpu)]
        else:
            contexts = [mx.context.cpu(i) for i in range(args.cpu)]

        init_states = lstm.init_states(hp.batch_size, hp.num_lstm_layer, hp.num_hidden)

        data_train = OCRIter(
            hp.train_epoch_size // hp.batch_size, hp.batch_size, init_states,hp.data_path,name='train')
        data_val = OCRIter(
            hp.eval_epoch_size // hp.batch_size, hp.batch_size, init_states,hp.data_path, name='val')

        symbol,seq_len = lstm.lstm_unroll(
            num_lstm_layer=hp.num_lstm_layer,
            num_hidden=hp.num_hidden,
            num_label=hp.num_label,
            loss_type=args.loss)

        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')

        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head)

        module = mx.mod.Module(
            symbol,
            data_names=['data', 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
            label_names=['label'],
            context=contexts)

        metrics = CtcMetrics(seq_len)
        module.fit(train_data=data_train,
                   eval_data=data_val,
                   eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
                   optimizer='sgd',
                   optimizer_params={'learning_rate': hp.learning_rate,
                                     'momentum': hp.momentum,
                                     'wd': 0.00001,
                                     },
                   initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                   arg_params=arg_params,
                   aux_params=aux_params,
                   num_epoch=hp.num_epoch,
                   batch_end_callback=mx.callback.Speedometer(hp.batch_size, 50),
                   epoch_end_callback=mx.callback.do_checkpoint(args.prefix,args.save_epoch),
                   )
    except KeyboardInterrupt:
        print("W: interrupt received, stopping...")


if __name__ == '__main__':
    main()

