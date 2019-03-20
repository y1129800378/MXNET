# CRNN_bucket

<a name="25b88ba9"></a>
### bucket意义
当进行OCR真实场景运用时，文字区域不可避免会出现不同尺寸和不同label大小。如果强行把训练数据resize成相同大小，则会出现严重形变，如果通过给图像补0的方式，补成相同大小，效果不一定好，
这时候就用到了bucket:支持不定长训练数据输入
<a name="d8b16075"></a>
### 训练
现在使用假数据做训练，数据为label长度为4，0~9的数据。 也可以识别汉字，英文等，只需对训练集做改变。
<a name="8c707d1c"></a>
#### 预训练模型
<br />训练数据地址: [传送门](https://pan.baidu.com/s/16Sq1b3zfPJveX45LBwDuPw) 提取码:5rwe
<a name="158744a8"></a>
#### 运行环境
* 运行环境： python2
* mxnet 1.2.0及以上
<a name="736947cb"></a>
#### 模型训练方法
<br />  先把下载的训练数据放在crnn_bucket/Train_data_bucket文件夹下<br /> 
```shell
$  python -u train.py --gpu 0 --loss ctc
```

CTC部分可以使用百度的waper CTC,也可以使用mxnet自带的CTC,建议使用自带CTC可以免去安装等步骤。



