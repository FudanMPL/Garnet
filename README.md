# Garnet

Garnet是复旦大学数据安全与治理研究组基于MP-SPDZ开发的一个实用化安全多方学习框架，支持SecureML协议、预训练模型的微调、XGBoost模型的安全训练。

## 部署
当前Garnet支持Linux 2014以及MacOS  High Sierra之后的操作系统版本。

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

## SecureML协议使用

SecureML是在Payman Mohassel和Yupeng Zhang发表于IEEE S&P'17的文章中提出的一个隐私保护机器学习框架。

其主要的特征为为线性回归、逻辑回归、神经网络等模型的训练过程设计了高效的安全计算协议；设计了定点数截断计算方案；引入了MPC友好的激活函数；引入了面向秘密共享的向量化计算。

在Garnet中SecureML框架对应的虚拟机是sml-party。sml-party虚拟机基于MP-SPDZ原生的semi-party和hemi-party。在两个参与方之间，将数据以加法秘密共享的形式分享。sml-party实现了基于OT的矩阵形式beaver三元组生成，和使用矩阵三元组的矩阵乘法和卷积操作。

### 基础设置
首次运行Garnet虚拟机时需要进行如下配置，如已成功运行过其他的两方虚拟机则可跳过此步。

安装必要的库

```
make -j8 tldr
```

 设置ssl

```
Scripts/setup-ssl.sh 2
```

###  编译运行sml-party虚拟机

以tutorial.mpc的测试程序为例

 设置输入、编辑mpc程序、设置环参数

```
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0
./compile.py -R 64 tutorial
```

 编译虚拟机

```
make -j 8 sml-party.x
```

 在两个终端分别运行

```
./sml-party.x -I 0 tutorial
./sml-party.x -I 1 tutorial
```

 或使用脚本

```
Scripts/sml.sh tutorial
```

## 运行预训练模型安全微调

  以LeNet和CK+48[1]数据集为例，若只希望载入自己的预训练模型进行安全微调，只需依次执行下列脚本即可：

1. 环境配置
首先修改CONFIG.mine文件，在开头加入如下一行代码。
```
MOD = -DRING_SIZE=32
```
之后，依次在控制台上输入以下三个命令进行虚拟机编译:
```
make clean
make -j 8 tldr
make -j 8 replicated-ring-party.x
```
下一步，在控制台上输入以下命令，生成证书及密钥
```
./Scripts/setup-ssl.sh 3
```
最后使用Script/ring.sh 运行样例程序(tutorial.mpc)，确保能够正常运行。

2. 获取预训练模型
```
python LeNet-Ferplus.py
```
3. 获取适用于Garnet的训练数据
```
python ./CK-plus-48-data-full.py
```
4. 编译安全微调的mpc文件
```
./compile.py -R 64 Lenet-fine-tuning.mpc
```
5. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
6. 在虚拟机中运行编译好的文件，进行微调
```
Scripts/ring.sh Lenet-fine-tuning  
```
  
若希望在Garnet中使用带有安全模型选择协议的安全微调，则可以通过执行下列脚本完成，在这里我们展示从两个预训练模型权重中选取最适合于CK+48数据集微调的权重的例子，权重对应的数据集分别为FER+[2]和CIFAR100[3]：


1. 获取一批预训练模型
```
python LeNet-Ferplus.py
python LeNet-CIFAR100.py
```
2. 获取适用于Garnet的训练数据
```
python ./CK+48-data-full.py
```
3. 获取预训练数据与用于微调的数据的平均特征
```
python ./VGG16-GetAll-Feature.py
```
4. 编译带有安全模型选择协议的安全微调的mpc文件
```
./compile.py -R 64 Lenet-fine-tuning-with-selection.mpc
```
5. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
6. 在虚拟机中运行编译好的文件，完成模型选择并进行微调
```
Scripts/ring.sh Lenet-fine-tuning-with-selection
```


[1] https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/tree/master/CK+48

[2] https://github.com/microsoft/FERPlus

[3] http://www.cs.toronto.edu/~kriz/cifar.html



## 运行XGBoost模型安全训练与预测
### 环境配置
首先修改CONFIG.mine文件（没有的话需要新建一个），在开头加入如下一行代码。
```
MOD = -DRING_SIZE=32
```

之后，依次在控制台上输入以下三个命令进行虚拟机编译:
```
make clean
make -j 8 tldr
make -j 8 replicated-ring-party.x
```

下一步，在控制台上输入以下命令，生成证书及密钥

```
./Scripts/setup-ssl.sh 3
```
最后使用Script/ring.sh 运行样例程序(tutorial.mpc)，确保能够正常运行。

### 数据准备
为了使用该算法进行训练，用户需要提供统一格式的训练数据并使用框架中所提供的脚本对数据进行处理。

用户首先需要在Data目录（没有的话需要新建一个）下存放csv格式的训练集和测试集。命名格式例如：训练集 IRIS_train.csv 测试集 IRIS_test.csv， 即使用 [数据集名]_train.csv 和 [数据集名]_test.csv 来命名。csv文件无需表头，每一行代表一个样本，最后一列代表标签。需要注意的是，本算法在读入小数时，会自动将小数部分截断，因此如果小数部分的数值有意义，请提前对小数部分乘上一个合适的系数并转化为整数。

在准备好csv格式的数据集后，运行python Script/data_prepare_for_xgboost [数据集名] 从而生成符合框架的数据格式，生成的文件为Player-Data/Input-P0-0。运行该脚本后，控制台会输出训练集所包含的训练样本数，特征数，测试集所包含的样本数，特征数。例如:
```
Garnet % python ./Scripts/data_prepare_for_decision_tree.py IRIS
file: ./Data/IRIS_train.csv
items: 135
attributes: 4
file: ./Data/IRIS_test.csv
items: 15
attributes: 4
```
### 脚本配置
在准备好上述数据后，根据需要修改Programs/xgboost.mpc。所需修改的信息为前六行。其中其三行根据上个步骤所输出的信息进行修改。

```
m = 4 # 特征数
n_train = 135 # 训练样本数量
n_test = 15 # 测试样本数量
h = 4 # 树高
n_estimators = 5 # 树的数量
n_threads = 4 # 最大线程数
```

### 运行
配置好上述脚本后，即可运行python ./compile.py xgboost -R 32获得编译后的二进制文件，该过程需要花费几分钟时间，且控制台显示的warning可以忽略。之后运行./Script/ring.sh xgboost 获得最后的运行结果。

```
Garnet % python ./compile.py xgboost -R 32
...
控制台输出
...
Writing to .../xgboost.sch
Writing to .../xgboost-0.bc
Program requires at most:
     3335730 integer inputs from player 0
    24960385 integer bits
    58387325 integer triples
         inf bit inverses
     3334980 integer inputs from player 1
       71918 virtual machine rounds
```

```

Garnet % ./Scripts/ring.sh xgboost
Using security parameter 40
Trying to run 32-bit computation
Training the 0-th tree
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
Training the 1-th tree
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
Training the 2-th tree
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
Training the 3-th tree
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
Training the 4-th tree
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
test for train set
true y = [0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 1, 2, 0, 2, 2, 0, 2, 2, 1, 1, 2, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 0, 1, 0, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
pred y = [0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 0, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 2, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 1, 2, 0, 2, 2, 0, 2, 2, 1, 1, 2, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 0, 1, 0, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0, 2, 2, 1, 2, 2, 0, 0, 0, 0, 1, 2]
pred y = [0, 0, 0.959335, 0.963852, 1.94325, 1.94325, 0.970718, 0.963852, 1.94325, 1.91367, 0, 0.963852, 0, 0.948578, 0, 1.93993, 0.963852, 0.0685577, 0, 0.963852, 0.292618, 0, 0.963852, 0.963852, 0.747147, 0.963852, 1.93993, 1.94325, 1.91367, 0, 0.959335, 1.93993, 0.963852, 1.93993, 0, 0.163681, 0, 0.963852, 0.962799, 0.963852, 0.963852, 0.0275421, 0.959335, 0.963852, 1.93335, 0, 0.92366, 0, 0, 0.941376, 1.93335, 0.963852, 0, 0.747147, 0, 0.0275421, 0.963852, 1.94325, 1.93993, 0.963852, 0.963852, 1.93993, 0.0977631, 1.93993, 1.9034, 0, 1.93993, 1.93993, 0.959335, 0.963852, 1.93335, 0.963852, 0, 0, 1.93993, 1.91313, 1.93993, 0, 1.94325, 0, 0, -0.0609131, 1.91646, 1.93993, 0.0977631, 0.963852, 0, 1.93993, 1.93335, 1.93993, 1.93993, 0.963852, 0.959335, 1.91492, 1.90637, 0.920074, 1.94022, 0.963852, 1.93993, 0.803986, 1.94325, 1.9097, 1.93993, 1.91678, 0.963852, 1.93993, 0, 0.970718, 0.94223, 0.963852, 1.94325, 0, 1.93993, 0.963852, 0.963852, 0, 1.9034, 0, 0, 0, 0.959824, 0, 1.94325, 0, 1.93993, 1.94325, 0.963852, 1.93993, 1.94325, 0, 0.194946, 0, 0, 0.963852, 1.93993] (not round)
accuracy: 135/135
test for test set
true y = [0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 1]
pred y = [0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 1]
pred y = [0, 0.0275421, 0, -0.0609131, 0.336411, 0.963852, 1.93993, 0, 0, 0, 0.970718, 0.920074, 0.949097, 1.93993, 1.04521] (not round)
accuracy: 15/15
Significant amount of unused dabits of replicated Z2^128. For more accurate benchmarks, consider reducing the batch size with -b.
Significant amount of unused dabits of replicated Z2^32. For more accurate benchmarks, consider reducing the batch size with -b.
Significant amount of unused dabits of replicated Z2^32. For more accurate benchmarks, consider reducing the batch size with -b.
Significant amount of unused dabits of replicated Z2^32. For more accurate benchmarks, consider reducing the batch size with -b.
Significant amount of unused dabits of replicated Z2^32. For more accurate benchmarks, consider reducing the batch size with -b.
Significant amount of unused dabits of replicated Z2^32. For more accurate benchmarks, consider reducing the batch size with -b.
The following benchmarks are including preprocessing (offline phase).
Time = 9.89406 seconds 
Data sent = 680.58 MB in ~100377 rounds (party 0; rounds counted double due to multi-threading)
Global data sent = 2041.73 MB (all parties)
This program might benefit from some protocol options.
Consider adding the following at the beginning of 'xgboost.mpc':
program.use_trunc_pr = True
program.use_split(3)
```
### 测试结果

|     |数据集名  | 训练集准确率 | 测试机准确率|
|  ----  | ----  |----  |----  |
|  1  | Kohkiloyeh | 93.33% | 80% |
|  2  | Diagnosis | 100% | 100% |
|  3  | IRIS | 100% | 100% |
|  4  | Wine | 100% | 94.44% |
|  5  | Cancer | 100% | 94.64% |
|  6  | Tic-tac-toe | 90.95% | 88.42% |

## 联系我们
如果您对项目有任何疑问，请在GitHub仓库上创建issue或者发送邮件到dsglab@fudan.edu.cn。
