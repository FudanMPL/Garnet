# <center> Garnet（石榴石） 
<div align=center><img width = '130' height ='130' src="./Garnet.png"/></div>
<br><br>

<p align="justify">Garnet（石榴石） 是继SecMML（Queqiao）后，由复旦大学数据安全与治理实验室开发并开源的又一个安全多方学习（MPL：Multi-party Learning）平台，其深度优化并扩展自MP-SPDZ（CCS 2020），并适配安全多方学习的特定功能需求。经过多轮迭代，Garnet是一个易用性高、通用性强且支持算法丰富的安全多方学习平台。当前，Garnet支持多种安全多方计算协议，例如：半诚实两方场景下的SecureML协议（Garnet新增）、任意多方不诚实大多数+恶意场景下的SPDZ协议等。用户可以使用Python脚本语言调用XGBoost（Garnet新增）模型安全训练功能以及预训练模型的安全微调（Garnet新增）功能。此外，Garnet还支持多场景（任意参与方数，诚实大多数+半诚实，不诚实大多数+半诚实等）下的逻辑回归、神经网络等机器学习模型的安全训练功能。</p>


## 从源码部署
当前Garnet支持Ubuntu 20.04以及MacOS  High Sierra之后的操作系统版本。

且Python版本需要 >= 3.10.0。

### 源码下载
```
git clone git@github.com:FudanMPL/Garnet.git
```

### 外部库准备

#### Linux
```
sudo apt-get install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4  texinfo yasm
```

#### MacOS

```
brew install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 texinfo yasm
```

### 初始编译（必须步骤）

```
make clean-deps boost libote
make clean
make -j 8 tldr
```


## 用Docker部署

### 源码下载
```
git clone git@github.com:FudanMPL/Garnet.git
```

### 创建镜像和容器

```
sudo docker build -t garnet .
```

```
sudo docker run --cap-add=NET_ADMIN -it garnet bash
```



## 历史发布

#### 2024年7月份发布: 

* [神经网络计算图优化](./docs/mpts.md)
* [安全高效的两方KNN协议](./docs/knn.md)
* [差分隐私随机梯度下降协议（论文PEA，S&P 2023）](./docs/dpsgd.md)
* [可配置特权方的n方安全多方计算协议](./docs/ruyi.md)
* [数据联合统计分析协议](docs/joint_statistics.md)
* [隐私集合求并](./docs/ss-psu.md)



#### 2024年1月份发布: 

* [GarnetUI](./GarnetUI/readme.md)
* [自动微分模块](./docs/autograd.md)
* [隐私集合求交](./docs/PSI.md)
* [安全字符串操作](./docs/string.md)
* [XGBoost训练扩展至两方场景和最多带有一个恶意参与方的三方场景](./docs/xgboost-training.md)

#### 2023年9月份发布: 

* [优化Function Secret Sharing的通信轮次](./docs/fss.md)
* [优化XGBoost推理所需的通信量](./docs/xgboost-inference.md)
* [优化向量空间秘密分享矩阵乘法离线阶段生成三元组的效率](./docs/vss.md)
* [新增Transformer模块及线性化ReLU层](./docs/transformer.md)





#### 2023年7月份发布: 

* [基于向量空间秘密共享的安全计算协议使用](./docs/vss.md)
* [Function Secret Sharing与Replicated Secret Sharing混合协议](./docs/fss.md)
* [基于NFGen的非线性函数近似计算](./docs/nfgen.md)
* [模型训练开销Profiling](./docs/profiling.md)
* [XGBoost模型安全推理](./docs/xgboost-inference.md)




#### 2023年3月份发布: 

* [SecureML协议](./docs/secureML.md)
* [预训练模型安全微调](./docs/pretrain.md)
* [XGBoost模型安全训练](./docs/xgboost-training.md)









## 联系我们
如果您对项目代码有任何疑问，请在GitHub仓库上创建issue。
