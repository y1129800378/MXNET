import mxnet as mx

def get_sym():
    data = mx.sym.Variable('data')
    conv1_1 = mx.symbol.Convolution(
            data=data, kernel=(3, 3), pad=(1, 1), num_filter=16, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
            data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=16, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
            data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    conv2_1 = mx.symbol.Convolution(
            data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=32, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
            data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=32, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
            data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    conv3_1 = mx.symbol.Convolution(
            data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
            data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
            data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
            data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
            pooling_convention="full", name="pool3")
    return pool3
