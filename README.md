# <center> Garnet（石榴石） 
<div align=center><img width = '130' height ='130' src="Garnet.jpeg"/></div>
<br><br>

<p align="justify">Garnet（石榴石） 是继SecMML（Queqiao）后，由复旦大学数据安全与治理研究组开发并开源的又一个安全多方学习（MPL：Multi-party Learning）框架，其深度优化并扩展自MP-SPDZ（CCS 2020），并适配安全多方学习的特定功能需求。相较前一个版本SecMML (Queqiao)，Garnet是一个高效、易用、功能多样的安全多方学习框架。当前，Garnet支持多种安全多方计算协议，例如：半诚实两方场景下的SecureML协议（Garnet新增）、任意多方不诚实大多数+恶意场景下的SPDZ协议等。用户可以使用类Python的脚本语言调用三方诚实大多数+半诚实安全模型场景下的XGBoost（Garnet新增）模型安全训练功能以及预训练模型的安全微调（Garnet新增）功能。此外，Garnet还支持多场景（任意参与方数，诚实大多数+半诚实，不诚实大多数+半诚实等）下的逻辑回归、神经网络等机器学习模型的安全训练功能。</p>


## 部署
当前Garnet支持Linux 2014以上以及MacOS  High Sierra之后的操作系统版本。

### 源码下载
```
git clone git@github.com:FudanMPL/Garnet.git
```

### 外部库准备

#### Linux
```
sudo apt-get install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm
```

#### MacOS

```
brew install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm
```

### 编译

```
make -j 8 tldr
```

## 历史发布

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
* [SecureML协议](./docs/secureML.md)







## 联系我们
如果您对项目有任何疑问，请在GitHub仓库上创建issue或者发送邮件到dsglab@fudan.edu.cn。
