# CRNN

<a name="25b88ba9"></a>
### crnn部分
crnn算法结构是CNN + 双向lstm + CTC . 本质是一个识别算法。即：CNN提取图像信息 ，lstm提取文字间的时序信息 ， 加CTC loss 。 CTC是为了解决文字识别时候不能逐像素对齐的问题。CTC在前传时候可以被softmax替代。
<a name="d8b16075"></a>
### 训练
现在使用假数据做训练，数据为label长度为4，0~9的数据。 也可以识别汉字，英文等，只需对训练集做改变。
<a name="8c707d1c"></a>
#### 训练数据有两种 
1. 定长输入（数据图像尺寸相同，label数相同）
1. 不定长输入（数据图像尺寸不同，label数不同

  注：目前开源项目为定长输入，而真实业务数据通常为不定长输入，需要使用bucket，这部分也会在不久开源。
<a name="a8f55deb"></a>
#### 预训练模型
<br />训练数据地址: [传送门](https://pan.baidu.com/s/1l2nqoPL2KI9HD-9nxQSPag) 提取码:350q
<a name="158744a8"></a>
#### 运行环境
* 运行环境： python2
* mxnet 1.2.0及以上
<a name="736947cb"></a>
#### 模型训练方法
step1:<br />  先把下载的训练数据放在crnn_fixed_length_input/Train_data文件夹，<br /> step2:<br />  预训练模型放在crnn_fixed_length_input/checkpoint文件夹（从0开始训练的可以忽略这步）   
step3:   训练
```shell
# 从0训练：
$  python -u train.py --gpu 0 --loss ctc

# 加载预训练模型进行训练
$ python -u train.py --gpu 0 --loss ctc --resume checkpoint/mobilenet,10 --prefix checkpoint/mobilnet
```

CTC部分可以使用百度的waper CTC,也可以使用mxnet自带的CTC,建议使用自带CTC可以免去安装等步骤。
