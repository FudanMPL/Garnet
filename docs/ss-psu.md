## 运行基于Secret Sharing 的多方隐私集合求并

多方隐私集合求并代码源码在文件`Compiler/library.py`中的ss_psu函数中

### 编译虚拟机

在控制台上输入以下命令进行虚拟机编译:

``` shell
make clean
make -j 8 tldr
make -j 8 semi2k-party.x
```

### 数据准备

用户需要在Player-Data/Input-P0-0下存放第0方的数据，在Player-Data/Input-P1-0下存放第1方的数据，以此类推。

这些文件的格式为，每行为一个数据样本，第一列为样本的ID，其他列为特征。

假设Player-Data/Input-P0-0中的数据为：

```
1
5
4
```


Player-Data/Input-P1-0中的数据为：

```
1
6
4
9
```

这些数据表明：第0方提供了3个样本。第1方提供了4个样本。

### 示例代码

已在Programs/Player-Data/test_psu中编写了示例代码：

```python
table0 = sint.Matrix(3, 1)
table1 = sint.Matrix(4, 1)

for x in table0:
    x.input_from(0) # 从第0方读入数据

for x in table1:
    x.input_from(1) # 从第1方读入数据

union_id, flag_id = ss_psu(table0, table1) # 进行PSU并合并两个数据表
```

该代码最终可以获得密态下的并集id（union_id），以及密态下的并集中指示id是否该被抛弃的标记（flag_id）。用户可以使用该表格进行下一步的运算。

### 编译
配置好上述脚本后，即可运行以下命令获得编译后的二进制文件。

```
python ./compile.py test_psu -R 64
```

### 运行

如果要运行两方半诚实场景下的PSU，则使用以下命令

```
./Scripts/semi2k.sh test_psu
```

我们将秘密共享的结果复原并打印，以便解释PSU返回的结果：

```
result = [0, 1]
result = [3, 9]
result = [1, 6]
result = [0, 4]
result = [-6736193989866728643, 1]
result = [3, 4]
```

这是复原后的结果，其中，第一维为0的代表其在交集中，我们需要在后续的处理中将其抛弃，余下的便是并集结果。注意，这些都是以秘密分享形式分发的，参与各方无法得知具体那条消息在并集中。
