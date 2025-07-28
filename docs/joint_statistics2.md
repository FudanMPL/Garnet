## 联合统计模块

联合统计（Joint Statistics）是多方安全计算（Multi-Party Computation, MPC）中的一个重要功能。它允许多个参与方在不泄露各自私有数据的情况下，共同计算出涉及这些数据的统计结果。联合统计可以应用于许多多方计算场景中。在本次更新中，我们为Garnet中增加了求标准差和方差的功能。

## 使用方法

其具体使用方法如下：


### 在mpc中调用相关的方法

此处a必须为元素类型为Array
```
a_mean = a.variance().reveal()
a_median = a.std_dev().reveal()
```

## 场景示例
    
编写以下mpc文件 其中a是元素类型为sfix的array，并对其进行求平均数，众数和中位数的操作

### test_joint_statistics代码
```
a = Array(10, sfix)
for i in range(10):
    if i < 5:
        a[i] = i + 1
    elif i == 8:
        a[i] = 7
    else:
        a[i] = 6
    print_ln("%s ", a[i].reveal())

a_square = a.square().reveal();
for i in range(10):
    print_ln("%s",a_square[i])
a_variance = a.variance().reveal();
print_ln("%s",a_variance)
a_std_dev = a.std_dev().reveal();
print_ln("%s",a_std_dev)
```

### 编译mpc文件
```
./compile.py -R 64 test_joint_statistics2
```


### 编译使用的虚拟机

```
make -j 8 sml-party.x
```
### 使用虚拟机运行mpc文件并查看结果
```
./sml-party.x -I 0 test_joint_statistics2
./sml-party.x -I 1 test_joint_statistics2
```
