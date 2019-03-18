CRNN部分
现在使用假数据做训练，数据为label长度为4，0~9的数据。
也可以识别汉字，英文等，只需对训练集做改变。

训练时训练数据有两种：
定长输入（数据图像尺寸相同，label数相同）
不定长输入（数据图像尺寸不同，label数不同）

目前开源项目为定长输入，而真实业务数据通常为不定长输入，需要使用bucket，这部分也会在不久开源



预训练模型，训练数据地址：
百度网盘：https://pan.baidu.com/s/1l2nqoPL2KI9HD-9nxQSPag
提取码：350q



模型训练方法：
先把下载的训练数据放在crnn_fixed_length_input/Train_data文件夹，预训练模型放在crnn_fixed_length_input/checkpoint文件夹（从0开始训练的可以忽略这步）
从0训练：
python -u train.py --gpu 0 --loss ctc

加载预训练模型进行训练：
python -u train.py --gpu 0 --loss ctc --resume checkpoint/mobilenet,10 --prefix checkpoint/mobilnet


CTC部分可以使用百度的waper CTC,也可以使用mxnet自带的CTC,建议使用自带CTC可以免去安装等步骤。



运行环境：
python2  
mxnet 1.2.0及以上

