

# MyImageCaptioningModel

使用PaddlePaddle实现的图片中文描述模型。使用AI_Challenger数据集。

![](https://camo.githubusercontent.com/004f23fee2877a76173477dc12ce8864ec7b8238/68747470733a2f2f73312e617831782e636f6d2f323032302f30362f31322f74584e7954302e6a7067)

预测结果： 一个 穿着 古装 的 女人 和 一个 穿着 白色 衣服 的 男人 坐 在 房间 里



# 依赖

-   PaddlePaddle 1.8
-   Numpy
-   h5py
-   pkuseg



# 使用方法

## 快速体验

您可以在百度[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/640590)中快速运行本项目。



## 安装

将本项目下载到本地即可

```
git clone git@github.com:Chgtaxihe/MyImageCaptioningModel.git
```



## 运行前设置

请在`config.py`中调整设置，部分设置项的说明如下

-   build_dataset: 构建数据集相关
    -   ImagePaths: AI_Challenger数据集的图像目录
    -   AnnotationPath: 数据集对应的标注文件
    -   OutputPath: 输出目录
    -   H5Name2Idx: 文件名-id索引文件，建议设置在OutputPath下
    -   sentence_len_limit: 句子最大长度
-   data: 训练时所使用的数据设置
    -   DictPath: npy文件所在目录（与`build_dataset`中`OutputPath`相同即可）
    -   H5Path: h5文件所在目录（与`build_dataset`中`OutputPath`相同即可）
    -   H5Name2Idx: 与`build_dataset`中`H5Name2Idx`相同即可
    -   sample_count: 训练集大小，根据所生成的数据集填写
-   train: 训练相关设置
    -   learning_rate: 学习率
    -   batch_size
    -   checkpoint_path: 检查点保存位置
    -   max_epoch: 最大epoch数
-   model: 模型参数
    -   'decoder': 
        -   vocab_size: 字典大小，请根据生成的数据集填写
        -   sentence_length: 训练所用的句子长度，请根据生成的数据集填写
        -   infer_max_length: 预测时句子的最大长度



## 构建数据集

在`config.py`中设置好`build_dataset`相关内容后，在控制台执行以下命令

```
python ./preprocess/dataset_gen.py
```

构建所需时间较长，请耐心等待。



## 训练

在开始训练之前，请确保您已经正确配置好`config.py`文件。

在控制台中执行以下命令即可

```
python ./train.py
```



## 执行预测

请将下方命令中的`url`替换为被预测图片的网址，并在控制台中执行。

```
python ./infer.py "url"
```



