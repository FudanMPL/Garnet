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

 设置输入、编译mpc程序、设置环参数

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