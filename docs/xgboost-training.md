## 运行XGBoost模型安全训练

本次更新后（2024年1月份更新），目前已支持的场景包括：两方半诚实场景、三方半诚实场景以及最多带有一个恶意参与方三方场景。

### 环境配置
首先修改CONFIG.mine文件（没有的话需要新建一个，请注意不是CONFIG文件），在开头加入如下一行代码。
```
MOD = -DRING_SIZE=32
```

之后，依次在控制台上输入以下命令进行虚拟机编译:
```
make clean
make -j 8 tldr
make -j 8 rss-with-conversion-party.x
make -j 8 mal-rss-with-conversion-party.x
make -j 8 semi2k-with-conversion-party.x
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

在准备好csv格式的数据集后，运行python Scripts/data_prepare_for_xgboost.py [数据集名] 从而生成符合框架的数据格式，生成的文件为Player-Data/Input-P0-0。运行该脚本后，控制台会输出训练集所包含的训练样本数，特征数，测试集所包含的样本数，特征数。例如:
```
python ./Scripts/data_prepare_for_xgboost.py IRIS

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

### 编译
配置好上述脚本后，即可运行python ./compile.py xgboost -R 32获得编译后的二进制文件，该过程需要花费几分钟时间，且控制台显示的warning可以忽略。

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

### 运行


如果要运行两方半诚实场景下的XGBoost训练，则使用以下命令

```
./Scripts/semi2k-with-conversion.sh xgboost
```

如果要运行三方半诚实场景下的XGBoost训练，则使用以下命令

```
./Scripts/rss-with-conversion.sh xgboost
```

如果要最多带有一个恶意参与方三方场景下的XGBoost训练，则使用以下命令

```
./Scripts/mal-rss-with-conversion.sh xgboost
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