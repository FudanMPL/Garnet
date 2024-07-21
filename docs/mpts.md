# 计算图优化模块

MPTS计算图优化模块是[TASO](https://github.com/jiazhihao/TASO)在MPL框架中的拓展，以通信量和通信轮次为优化目标优化输入的计算图，在不影响计算结果的条件下优化输入的模型结构。

## 功能介绍

《TASO: The Tensor Algebra SuperOptimizer for Deep Learning》是Zhihao Jia等发表于SOSP 2019上的文章，使用自动生成的子图转换规则来搜索与原始DNN模型等价的潜在搜索空间中的计算图。转换规则由一组等价的源图和目标图组成，两者均为DNN算子连接成的1-3层深度的有向无环图，计算图优化通过迭代地匹配并应用转换规则可发现更大的搜索空间。MPTS计算图优化模块继承了TASO中的134条优化规则，拓展了基于通信量和通信轮次的开销模型，并实现了考虑通信成本和浮点数计算次数的搜索减枝算法，提高了其搜索的收敛速度。

## 环境配置

安装并配置如下依赖：
- CMAKE 3.2
- ProtocolBuffer 3.6.1
- Cython 0.28
- CUDA 11.7 and CUDNN 7.0

并在 `~/.bashrc` 配置Garnet环境路径。
```
export TASO_HOME=/path/to/MPLTS
export Garnet_HOME=/path/to/Garnet
export LD_LIBRARY_PATH="/path/to/lib/:$LD_LIBRARY_PATH"
```
从`requirements.txt`中安装额外python依赖。
```
pip install -r MPLTS/requirements.txt
```
编译并安装MPLTS库。
```
./MPLTS/recompile.sh
```


## 运行优化样例
目前MPLTS支持了基本的卷积块和激活函数，以resnet18和resnet50的单个块为例优化模型结构。在实验中我们尝试SNL和Copriv论文中的两类线性化的卷积神经网络进行优化。
```
python MPLTS/examples/resnet18.py -R 64 -Q ABY3
python MPLTS/examples/resnet50.py -R 64 -Q ABY3
```

优化效率
| 模型    | 通信量优化 | 通信轮次优化 | 总通信量优化 | 总通信轮次优化 |
| --------- | ------ | -------- | -------- | ---------- |
| ResNet-18 | 1.744x | 1.013x   | 1.295x   | 1.080x     |
| ResNet-50 | 1.337x | 1.011x   | 1.107x   | 1.497x     |


## 使用优化模型推理
首先模型载入Garnet框架。
```
from Compiler.onnxConverter.model import 
ConvertModel
import onnx
onnx_model = onnx.load("example.onnx")
model = ConvertModel(onnx_model)
print(model)
```
初始化输入x后用模型推理得到推理结果y。

```
x = MultiArray([1,32,10,10], sfix)
@for_range(x.total_size())
def _(i):
    x.assign_vector(sfix(i/10), i)
input = Tensor(x)
y= model(input)
```



## 正确性验证
首先需要构建虚拟机运行所需的[Garnet运行环境](https://github.com/FudanMPL/Garnet?tab=readme-ov-file#%E5%88%9D%E5%A7%8B%E7%BC%96%E8%AF%91%E5%BF%85%E9%A1%BB%E6%AD%A5%E9%AA%A4), 以三方复制秘密共享为例。

```
make -j 8 tldr
make -j 8 replicated-ring-party.x
make -j 8 Fake-Offline.x
./Scripts/setup-ssl.sh 3
```

在Garnet上对优化结构的正确性进行验证。将分次输出明文推理结果、隐私推理结果、优化后的隐私推理结果。

```
python MPLTS/Scripts/check.py -R 64
```

以`resnet_block`的推理结果为例，计算误差并验证。
```
python MPLTS/Scripts/check_fig
```
