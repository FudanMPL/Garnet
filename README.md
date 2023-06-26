# <center> Garnet（石榴石） 
<div align=center><img width = '130' height ='130' src="fig/Garnet.jpeg"/></div>
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

## SecureML协议使用

SecureML是在Payman Mohassel和Yupeng Zhang发表于IEEE S&P'17的文章中提出的一个隐私保护机器学习框架。

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

### 环境配置

在主目录（/Garnet）下提供了requirements.txt和fine-tuning.yaml用于配置环境，可以在主目录下执行以下命令完成环境配置

```
pip install -r ./requirements.txt 
```

### 数据准备
生成的模型和数据可以从https://drive.google.com/file/d/1OOBGQroO4YmlBAPqEbz1VUHNiKBTDXGB/view?usp=sharing 得到。其中数据来自于[1] , [2]和[3]。下载后解压将三个文件夹及其内容拷贝至/Garnet/Compiler/DL目录下即可


### 代码运行
  以LeNet和[CK+48][1]数据集为例，若只希望载入自己的预训练模型进行安全fine-tuning，只需依次执行下列脚本即可：


1. 获取预训练模型，在Garnet主目录运行如下命令
```
cd Compiler/DL
python LeNet-Ferplus.py
```
2. 获取适用于Garnet的训练数据
```
python ./CK+48-data-full.py
```
3. 编译安全fine-tuning的mpc文件
```
cd ../../
python compile.py -R 64 torch_lenet_fine-tuning
```
4. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
5. 在虚拟机中运行编译好的文件，进行fine-tuning
```
Scripts/ring.sh torch_lenet_fine-tuning  
```

  若希望在Garnet中使用带有安全模型选择协议的安全fine-tuning，则可以通过执行下列脚本完成，在这里我们展示从两个预训练模型权重中选取最适合于CK+48数据集fine-tuning的权重的例子，权重对应的数据集分别为[FER+][2]和[CIFAR100][3]：


1. 获取一批预训练模型,，在Garnet主目录运行如下命令
```
cd ./Compiler/DL
python LeNet-Ferplus.py
python LeNet-CIFAR100.py
```
2. 获取适用于Garnet的训练数据
```
python ./CK+48-data-full.py
```
3. 获取预训练数据与用于fine-tuning的数据的平均特征
```
python ./GetAll-feature.py
```
4. 编译带有安全模型选择协议的安全fine-tuning的mpc文件
```
cd ../../
python compile.py -R 64 torch_ckplus48_lenet_selected
```
5. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
6. 在虚拟机中运行编译好的文件，完成模型选择并进行fine-tuning
```
Scripts/ring.sh torch_ckplus48_lenet_selected
```


[1]: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/tree/master/CK+48
[2]: https://github.com/microsoft/FERPlus
[3]: http://www.cs.toronto.edu/~kriz/cifar.html




## 运行XGBoost模型安全训练与预测
### 环境配置
首先修改CONFIG.mine文件（没有的话需要新建一个，请注意不是CONFIG文件），在开头加入如下一行代码。
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



### 数据准备
为了使用该算法进行训练，用户需要提供统一格式的训练数据并使用框架中所提供的脚本对数据进行处理。

首先确保本地的python安装了pandas库，否则使用以下命令安装

```
pip install pandas
```

用户需要在Data目录（没有的话需要新建一个）下存放csv格式的训练集和测试集。命名格式例如：训练集 IRIS_train.csv 测试集 IRIS_test.csv， 即使用 [数据集名]_train.csv 和 [数据集名]_test.csv 来命名。csv文件无需表头，每一行代表一个样本，最后一列代表标签。需要注意的是，本算法在读入小数时，会自动将小数部分截断，因此如果小数部分的数值有意义，请提前对小数部分乘上一个合适的系数并转化为整数。

这里我们提供了符合格式的IRIS数据集的下载地址：https://drive.google.com/drive/folders/1dLUA7VRHGNkvpH7cgIPIsRqLvO4nABb8?usp=sharing

在准备好csv格式的数据集后，运行python Scripts/data_prepare_for_xgboost [数据集名] 从而生成符合框架的数据格式，生成的文件为Player-Data/Input-P0-0。运行该脚本后，控制台会输出训练集所包含的训练样本数，特征数，测试集所包含的样本数，特征数。例如:
```
python ./Scripts/data_prepare_for_decision_tree.py IRIS

以下为控制台输出
file: ./Data/IRIS_train.csv
items: 135
attributes: 4
file: ./Data/IRIS_test.csv
items: 15
attributes: 4
```
### 脚本配置
在准备好上述数据后，根据需要修改Programs/Source/xgboost.mpc。所需修改的信息为前六行。其中其三行根据上个步骤所输出的信息进行修改。

```
m = 4 # 特征数
n_train = 135 # 训练样本数量
n_test = 15 # 测试样本数量
h = 4 # 树高
n_estimators = 5 # 树的数量
n_threads = 4 # 最大线程数
```

### 运行
配置好上述脚本后，即可运行python ./compile.py xgboost -R 32获得编译后的二进制文件，该过程需要花费几分钟时间，且控制台显示的warning可以忽略。之后运行./Scripts/ring.sh xgboost 获得最后的运行结果。

```
python ./compile.py xgboost -R 32


以下为控制台输出
......
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

./Scripts/ring.sh xgboost

以下为控制台输出
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

最后我们提供使用上述的参数（树高为4，树的数量为5)在以下数据集中的测试结果

|     |数据集名  | 训练集准确率 | 测试集准确率|
|  ----  | ----  |----  |----  |
|  1  | Kohkiloyeh | 93.33% | 80% |
|  2  | Diagnosis | 100% | 100% |
|  3  | IRIS | 100% | 100% |
|  4  | Wine | 100% | 94.44% |
|  5  | Cancer | 100% | 94.64% |
|  6  | Tic-tac-toe | 90.95% | 88.42% |

## 向量空间秘密共享技术使用

在Garnet中向量空间秘密共享技术所对应的虚拟机是vss-party。vss-party基于MP-SPDZ原生的semi-party和hemi-party。vss-party实现了基于向量空间秘密共享的三方安全乘法、安全截断、安全比较等操作。

### 编译运行vss-party虚拟机

以tutorial.mpc的测试程序为例

设置输入、编辑mpc程序、设置环参数

```
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0
echo 1 2 3 4 > Player-Data/Input-P2-0
./compile.py -R 64 tutorial
```

 编译虚拟机

```
make -j 8 vss-party.x
```

 在两个终端分别运行

```
./vss-party.x -I 0 tutorial
./vss-party.x -I 1 tutorial
./vss-party.x -I 2 tutorial
```

 或使用脚本

```
Scripts/vss.sh tutorial
```

## 联系我们
如果您对项目有任何疑问，请在GitHub仓库上创建issue或者发送邮件到dsglab@fudan.edu.cn。
