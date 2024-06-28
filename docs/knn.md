##  运行KNN算法安全训练

**模型介绍：**两方半诚实场景，Client-Server架构下的基于秘密共享技术的KNN（K-Nearest Neighbor）算法。

利用Garnet系统现有的一些接口，本文实现了一个两方下隐私保护的KNN算法。为方便代码实现和阅读，本文使用了固定的场景：一个参与方P0拥有训练数据集**D**，P1拥有一个无标签的特征向量**q**，P0和P1共同在P0数据集**D**中运行隐私保护的KNN算法来获得**q**的分类结果，最终达到P1无法获取到除了预测结果之外的其他信息，数据集**D**的拥有者P0也无法获取到包括预测结果等任何信息,本文代码默认在环上进行运算。

在本文中，针对两方下基于加法秘密共享技术的隐私保护KNN算法实现，给出了两种实现方案，分别为 [SecKNN,2024,TIFs](https://ieeexplore.ieee.org/document/10339363/footnotes#footnotes)
论文的实现方案以及本文针对该方案的优化版本，该优化在当前的数据集测试中均取得了符合实验预期的实验结果。
### 输入文件格式规范
下面针对Chronic KIdney Disease dataset 数据集进行举例子。首先进行数据的预处理，将原始数据集归一化到[0,10]的整数范围内。按照70%划分为训练集，30%划分为测试集。然后需要在/Player-Data目录下创建一个Knn-Data目录，然后针对当前数据集创建chronic-data目录，随后在该目录下面按照要求分别创建如下文件，并指定文件内容。下面对文件内容举例并解释对应功能：
#### /Player-Data/Knn-Data/chronic-data/Knn-meta：
```markdown
24 280 120
2
0 10
```
第一行三个数据，分别为训练集的特征向量的个数，P0的训练集D的数据条数，P1需要查询的无标签数据的向量个数。
第二行为训练集的类别个数。
第三行对应的类别的两个值，分别为0,10。

#### ./Player-Data/Knn-Data/chronic-data/P0-0-X-Train：
```markdown
8 2 3 6 0 10 0 10 10 4 2 0 8 0 5 5 1 3 10 10 10 0 0 0
8 3 3 0 6 10 10 0 0 6 1 0 8 0 7 6 3 4 10 10 0 0 0 0
4 4 3 0 0 0 10 0 0 2 0 0 8 0 8 8 1 5 10 0 0 0 0 0
5 3 5 2 0 0 0 0 0 1 1 0 9 1 5 7 1 5 0 0 0 0 10 0
7 2 8 0 0 10 10 0 0 2 1 0 9 1 7 9 2 5 0 0 0 0 0 0
...
```
数据维度为280 x 24，每一行为一个样本的特征向量，24个特征向量数据。
#### ./Player-Data/Knn-Data/chronic-data/P0-0-Y-Train:
```markdown
0
0
0
0
10
...
```
数据维度为280 x 1，每一行表示./Player-Data/Knn-Data/chronic-data/P0-0-X-Train文件中对应行数据的标签。
#### ./Player-Data/Knn-Data/chronic-data/P1-0-X-Test:
```markdown
5 2 10 0 0 10 10 0 0 4 4 2 8 0 4 3 5 1 10 0 0 0 10 10
6 1 8 0 0 10 10 0 0 2 1 0 8 1 10 7 2 6 0 0 0 0 0 0
...
```
该文件为参与方P1准备的测试文件，文件数据维度为120 x 24，每一行表示一个无标签的特征向量。
#### ./Player-Data/Knn-Data/chronic-data/P1-0-Y-Test:
```markdown
0
10
10
0
...
```
该文件为参与方P1准备的测试文件对应的标签，用于测试算法的准确率。


这里我们提供了符合格式的当前数据集的下载地址：[https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing](https://drive.google.com/drive/folders/1YhiUd44POknGhOE1NP36voe9RmeHwWHr?usp=sharing)

下载后放入Player-Data目录下即可。


### 代码运行流程
（1） 进入Garnet目录，在命令行界面中输入以下命令进行虚拟机编译:
```markdown
make clean
make -j 8 tldr
```
（2） 进行offline数据三元组以及函数秘密共享需要的离线数据的生成。为了方便生成，本文采用Dealer第三方生成模式，使用如下指令编译并生成对应的代码：
```markdown
make -j 8 knn-party-offline.x//编译knn-party-offline.cpp文件
./knn-party-offline.x //生成对应需要的离线数据
```

（3） 在控制台上输入以下命令，生成证书及密钥
```markdown
./Scripts/setup-ssl.sh 2
```
（4） 如果是本地测试，可以直接打开两个命令行窗口，在编译knn-patry.x文件后，分别运行以下两个命令：
```markdown
make -j 8 knn-patry.x//编译knn-patry.x文件
./knn-party.x 0 -pn 11126 -h localhost //模拟参与方P0运行的命令
./knn-party.x 1 -pn 11126 -h localhost//模拟参与方P1运行的命令
```
（5）实验结果：
```markdown
测准确率：0.966667
Total Round count = 13440 online round
Party total time = 77.4452 seconds
Party Data sent = 12.0134 MB
call_evaluate_nums : 337440
在Evaluation函数中 Total elapsed time: 76.0757 seconds
```

### 指定参数
如果需要自己指定数据集，以及计算环大小，KNN算法中k值大小，需要在Machines/knn-party.cpp代码文件中进行修改，寻找如下代码段进行修改，然后使用上面的命令进行编译运行即可。
```markdown
const int K=64;//环大小
const int k_const=5;//knn里面的k值 
string dataset_name="chronic";//数据集名称，自动用于后续的文件名生成
```
