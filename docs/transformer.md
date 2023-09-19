# Transformer和线性化ReLU层

## 更新介绍

本次更新增加了LayerNorm层，SoftMax层，Linear层以及基于这些层实现的Transformer基本模块。此外本次更新还增加了安全推理方法中常用的线性化ReLU层，该层可用于实现低通讯开销的线性化神经网络

## Transformer基本块

1. 运行MNIST-ViT.py生成测试数据，并将生成的数据直接拷贝到Player-Data中

```
cd Compiler/DL
python MNIST-ViT.py
cp ./Input-P0-0 ../../Player-Data/Input-P0-0
```

2. 回到主目录，编译并运行ViT-MNIST-Test.mpc

```
cd ../../
python compile.py -R 64 ViT-MNIST-Test
Scripts/ring.sh ViT-MNIST-Test
```

## 线性化ReLU层

1. 在主目录编译并运行Linear-ReLU-Test.mpc

```
python compile.py -R 64 Linear-ReLU-Test
Scripts/ring.sh Linear-ReLU-Test
```
