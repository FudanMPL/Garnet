# 计算图优化模块

MPTS计算图优化模块是[TASO](https://github.com/jiazhihao/TASO)在MPL框架中的拓展，以通信量和通信轮次为优化目标优化输入的计算图，在不影响计算结果的条件下进行模型结构的等价变换。

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
以resnet和nasnet的单个块为例优化模型结构。
```
python MPLTS/examples/resnet_block.py -R 64 -Q  ABY3
python MPLTS/examples/nasnet_block.py -R 64 -Q  ABY3
```

## 使用优化模型推理
首先模型载入Garnet框架。
```
from Compiler.Convert.model import 
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
