##  安全高效的两方Ann协议

**模型介绍**:两方半诚实场景，Client-Server架构下的基于秘密共享技术的ANN（Approximate Nearest Neighbor）算法。  

利用Garnet系统接口，本文实现了一个两方下隐私保护的ANN协议。本文实现的场景为多个数据拥有者将本地隐私数据进行预处理之后，通过加法秘密共享协议进行秘密分享给两个互不合谋的服务器，服务器运行安全的ANN协议，当数据查询者需要查询时，发送给服务器秘密共享状态的查询向量数据，服务器在保护多个数据拥有者的数据隐私的情况下，得到最近邻的近似数据集，返回给数据查询者。


### 输入文件格式规范

下面针对 Mnist 数据集进行举例。首先进行数据预处理，将原始数据集归一化到[0,10]的整数范围内。然后需要在/Player-Data目录下创建一个Ann-Data目录，针对当前数据集创建 Mnist 目录。随后在该目录下面按照要求分别创建如下文件，并指定文件内容。

#### /Player-Data/Ann-Data/Mnist-data/Ann-meta：
```markdown
784 69999 1
784 69999 1
```
三个数据分别为数据集的特征向量的个数，数据集中的数据条数(即行数)，需要查询的测试数

#### ./Player-Data/Ann-Data/Mnist-data/P0-0-X-Train：
```markdown
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
....
....
...
```
该文件为数据拥有者的数据文件，内容为数据集的特征向量。维度为69999 x 784 ，每一行一条数据的特征向量，每条数据有784个特征向量元素。以上维度数分别与 Ann-meta 文件数值对应。
#### ./Player-Data/Ann-Data/chronic-data/P0-0-Y-Train:
```markdown
9
4
9
5
6
1
3
1
3
...
```
该文件为参与方P0的输入训练集数据文件，内容为训练集的标签值。数据维度为69999 x 1，每一行的数据与./Player-Data/Ann-Data/Mnist-data/P0-0-X-Train文件中对应数据集的每一行数据的标签一一对应。以上维度数分别与 Ann-meta 文件数值对应。
#### ./Player-Data/Ann-Data/Mnist-data/P1-0-X-Test:
```markdown
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
...
```
该文件为参与方P1的输入测试集文件，内容为测试集的特征向量。每一行的数据表示一个无标签的特征向量。以上维度数分别与 Ann-meta 文件数值对应。
#### ./Player-Data/Ann-Data/Mnist-data/P1-0-Y-Test:
```markdown
0
```
该文件为参与方P1的输入测试集文件对应的标签，用于测试该安全ANN算法的预测准确率。


**注：**
为方便直接测试使用，这里我们提供了符合上述数据格式规范的两个数据集的下载地址：[https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing](https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing)
下载后放入Player-Data目录下后按照下面**代码运行流程**章节，从（2）开始运行即可。


### 代码运行流程
（1） 源码下载，以及相应系统的外部库准备，随后进入Garnet目录，在命令行界面中输入以下命令编译虚拟机:
```markdown
#源码下载
git clone https://github.com/FudanMPL/Garnet.git --recursive #源码下载
#Linux外部库准备
sudo apt-get install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4  texinfo yasm
#MacOS外部库准备
brew install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 texinfo yasm
make clean  
make -j 8 tldr
```
（2）准备输入数据，并放入到Player-Data目录下,创建Knn-Data，下载并放入对应文件。具体输入数据规范见前面**输入文件格式规范**章节，或者直接从数据集下载地址下载，即[https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing](https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing)。

（3） 在控制台上输入以下命令，生成证书及密钥
```markdown
./Scripts/setup-ssl.sh 2
```
（4） 如果是本地测试，使用如下命令进行编译和运行K-means文件，以及编译生成可执行文件Ann-party.x 。
```markdown
g++ -O3 -fopenmp -march=native ./Machines/K-means.cpp -o Kmeans
./Kmeans
make -j 8 ANN.x  #编译ANN.cpp文件
```
 (5)  本地运行：打开两个命令行窗口，分别在两个命令行窗口中运行以下命令：
```markdown
./ANN.x 0 -pn 10000 -h localhost  #第一个命令行窗口运行该指令，模拟参与方P0运行的命令
./ANN.x 1 -pn 10000 -h localhost  #第二个命令行窗口运行该指令，模拟参与方P1运行的命令
```
```
（6）获得实验结果：
```markdown
Best k selected by elbow method: 9

Cluster_0:6430
Cluster_1:8494
Cluster_2:7724
Cluster_3:6269
Cluster_4:5859
Cluster_5:5602
Cluster_6:12141
Cluster_7:9292
Cluster_8:8188
Selected Cluster:4
Saved Kmeans_Result to KmeansRes


--------DataSet:Mnist--------------
Entering the Ann_party_optimized class:
Network Set Up Successful ! 
sample size:5859
test size:1
Feature size:784
num_train_data:5859

预测准确率:1
Total Round count = 138 online round
Party total time = 4.5379 seconds
Party Data sent = 2.10905 MB
call_evaluate_nums : 58618
在Evaluation函数中 Total elapsed time: 4.48537 seconds
-----------------------------------------------------

```
可以看到在获取到查询向量最近邻的聚类编号之后，对该编号的聚类簇运行安全KNN协议运算，最后得到最近邻的k个向量数据以及分类结果。
可以看到，通过ANN算法，可以得到聚类簇个数倍的效率加速，在Mnist数据集中，对于69999个数据在引入聚类算法之后减少到某个簇，该簇中需要KNN查询的数据条数缩减为5859条，极大的提升了最终的查询速度。

### 指定参数以及运行方案
（1）如果需要自己指定数据集，以及计算环大小，Ann协议中k值大小，需要在Machines/K-means.cpp代码文件中进行修改，寻找如下代码段进行修改，按照字符串增加或者减少数据集：
```markdown
vector<string>dataset_name_list={"Mnist"};
```
在Machines/ANN.cpp代码文件中进行修改，寻找如下代码段，进行修改
```markdown
const int K=64;//环大小
const int k_const=5;//Ann里面的k值 
vector<string>dataset_name_list={"Mnist"};//数据集名称，自动用于后续的文件名生成
```

