##  两方隐私保护KNN算法

**模型介绍**:两方半诚实场景，Client-Server架构下的基于秘密共享技术的KNN（K-Nearest Neighbor）算法。

利用Garnet系统接口，本文实现了一个两方下隐私保护的KNN算法。本文使用了固定的场景：参与方P0拥有训练数据集**D**，参与方P1拥有多个无标签的特征向量**Q**，P0和P1共同在数据集**D**中运行隐私保护的KNN算法来获得**Q**中每一条数据的分类结果。可以保证参与方P1无法获取到除了预测结果之外的其他信息，参与方P0无法获取到包括预测结果在内的任何信息。所有运算默认在环上进行。

本文针对两方基于加法秘密共享技术的隐私保护KNN算法实现，给出了两种实现方案，分别为 论文[SecKNN,2024‘TIFs](https://ieeexplore.ieee.org/document/10339363/footnotes#footnotes)
的方案以及本文针对该论文的通信效率优化版本，优化方案在当前的数据集测试中均取得了符合实验预期的实验结果。
### 输入文件格式规范
下面针对Chronic KIdney Disease dataset 数据集进行举例子。首先进行数据预处理，将原始数据集归一化到[0,10]的整数范围内。按照70%划分为训练集，30%划分为测试集。然后需要在/Player-Data目录下创建一个Knn-Data目录，针对当前数据集创建chronic-data目录。随后在该目录下面按照要求分别创建如下文件，并指定文件内容。
#### /Player-Data/Knn-Data/chronic-data/Knn-meta：
```markdown
24 280 120
2
0 10
```
第一行三个数据，分别为训练集的特征向量的个数，P0的训练集**D的**数据条数(即行数)，P1需要查询的无标签数据**Q**的向量个数(即行数)。
第二行为训练集的类别个数。
第三行对应的类别的值，分别为0和10。

#### ./Player-Data/Knn-Data/chronic-data/P0-0-X-Train：
```markdown
8 2 3 6 0 10 0 10 10 4 2 0 8 0 5 5 1 3 10 10 10 0 0 0
8 3 3 0 6 10 10 0 0 6 1 0 8 0 7 6 3 4 10 10 0 0 0 0
4 4 3 0 0 0 10 0 0 2 0 0 8 0 8 8 1 5 10 0 0 0 0 0
5 3 5 2 0 0 0 0 0 1 1 0 9 1 5 7 1 5 0 0 0 0 10 0
7 2 8 0 0 10 10 0 0 2 1 0 9 1 7 9 2 5 0 0 0 0 0 0
...
```
该文件为参与方P0的输入训练集数据文件，内容为训练集的特征向量。维度为280 x 24，每一行一条数据的特征向量，每条数据有24个特征向量元素。以上维度数分别与Knn-meta文件数值对应。
#### ./Player-Data/Knn-Data/chronic-data/P0-0-Y-Train:
```markdown
0
0
0
0
10
...
```
该文件为参与方P0的输入训练集数据文件，内容为训练集的标签值。数据维度为280 x 1，每一行表示./Player-Data/Knn-Data/chronic-data/P0-0-X-Train文件中对应行数据的标签。以上维度数分别与Knn-meta文件数值对应。
#### ./Player-Data/Knn-Data/chronic-data/P1-0-X-Test:
```markdown
5 2 10 0 0 10 10 0 0 4 4 2 8 0 4 3 5 1 10 0 0 0 10 10
6 1 8 0 0 10 10 0 0 2 1 0 8 1 10 7 2 6 0 0 0 0 0 0
...
```
该文件为参与方P1的输入测试集文件，内容为测试集的特征向量。文件数据维度为120 x 24，每一行的数据表示一个无标签的特征向量。以上维度数分别与Knn-meta文件数值对应。
#### ./Player-Data/Knn-Data/chronic-data/P1-0-Y-Test:
```markdown
0
10
10
0
...
```
该文件为参与方P1的输入测试集文件对应的标签，用于测试算法的准确率。文件数据维度为120 x 1。维度数分别与Knn-meta文件数值对应。


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
（2）准备输入数据，并放入到Player-Data目录下。具体输入数据规范见前面**输入文件格式规范**章节，或者直接从数据集下载地址下载，即[https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing](https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing)。

（3） offline数据三元组以及函数秘密共享需要的离线数据的生成。为了方便实现，本文采用可信第三方生成模式，用户可以直接使用如下指令编译并运行对应的Machine/knn-party-offline.cpp代码：
```markdown
make -j 8 knn-party-offline.x #编译knn-party-offline.cpp文件
./knn-party-offline.x  #生成对应需要的离线数据
```

（4） 在控制台上输入以下命令，生成证书及密钥
```markdown
./Scripts/setup-ssl.sh 2
```
（5） 如果是本地测试，可以直接打开两个命令行窗口，在编译knn-patry.x文件后，分别在两个命令行窗口中运行以下命令：
```markdown
make -j 8 knn-party.x  #编译knn-patry.x文件
./knn-party.x 0 -pn 11126 -h localhost  #第一个命令行窗口运行该指令，模拟参与方P0运行的命令
./knn-party.x 1 -pn 11126 -h localhost  #第二个命令行窗口运行该指令，模拟参与方P1运行的命令
```
（6）实验结果：
```markdown
测准确率：0.966667
Total Round count = 13440 online round
Party total time = 77.4452 seconds
Party Data sent = 12.0134 MB
call_evaluate_nums : 337440
在Evaluation函数中 Total elapsed time: 76.0757 seconds
```
可以看到在优化版本中，大部分时间为本地运算开销，通信开销在被优化后，基本可以忽略。

### 指定参数以及运行方案
（1）如果需要自己指定数据集，以及计算环大小，KNN算法中k值大小，需要在Machines/knn-party.cpp代码文件中进行修改，寻找如下代码段进行修改：
```markdown
const int K=64;//环大小
const int k_const=5;//knn里面的k值 
string dataset_name="chronic";//数据集名称，自动用于后续的文件名生成
```
除此之外，还需要在Machines/knn-party-offline.cpp代码文件中进行修改，寻找如下代码段进行修改：
```markdown
string dataset_name="chronic";//数据集名称，自动用于后续的文件名生成
```
随后，按照前面**代码运行流程**章节，从（3）开始运行即可。


（2）如果需要指定运行论文[SecKNN,2024‘TIFs](https://ieeexplore.ieee.org/document/10339363/footnotes#footnotes)
的实现方案，只需要在Machines/knn-party.cpp代码文件的main函数中修改为如下代码：
```markdown
int main(int argc, const char** argv)
{
    parse_argv(argc, argv);
    //KNN_party_optimized party(playerno);
    KNN_party_SecKNN party(playerno);
    party.start_networking(opt);
    std::cout<<"Network Set Up Successful ! "<< std::endl;
    party.run();
    return 0;
}
```
然后根据前面**代码运行流程**章节，从（5）开始运行即可。
在（6）步骤中可以得到如下的实验结果
```markdown
预测准确率 : 0.966667
Total Round count = 335160 online round
Party total time = 82.6868 seconds
Party Data sent = 18.4646 MB
call_evaluate_nums : 337440
在Evaluation函数中 Total elapsed time: 75.653 seconds
```
