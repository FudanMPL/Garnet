## 可配置特权方的n方安全多方计算协议

可配置特权方的n方安全多方计算协议 利用了基于域的向量空间秘密共享的安全计算协议进行实现，支持任意多方。

### 功能介绍

向量空间秘密共享（Vector Space Secret Shaing）是Song等人发表于ACM CCS‘22 的安全多方学习框架pMPL所使用的底层秘密共享技术，公共矩阵的约束较多导致可配置性有限。

安全多方学习框架Ruyi在保证相同特权保证的同时，将公开矩阵的约束从四个减少到两个，并在素数域而不是环上执行所有的计算。这增强了可配置性，使得范德蒙德矩阵在给定的参与方的数量下始终满足公共矩阵限制，包括特权方、辅助方和允许掉线的辅助方数量。

在Garnet中基于域的向量空间秘密共享技术所对应的虚拟机是vss-field-party，实现了基于域上的向量空间秘密共享的多方安全计算操作，可扩展至任意多方，用户在运行时可配置特权方数量NP、协助方数量NA、可掉线的协助方数量ND。

接下来以三方为例，展示vss-field-party虚拟机的使用过程。

### 基础设置

设置ssl

```
Scripts/setup-ssl.sh 3
```

### 编译tutorial程序

设置输入、编译mpc程序

```
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0
echo 1 2 3 4 > Player-Data/Input-P2-0
./compile.py tutorial
```

### 编译vss-field-party虚拟机

```
make -j 8 vss-field-party.x
```

### 运行vss-party虚拟机

ND用于配置可掉线的协助方数量，NA用于配置协助方数量，NP用于配置特权方数量

总参与方数量是NP、NA之和，总终端数量是NP、NA、ND之和

在三个终端分别运行

```
./vss-field-party.x 0 -ND 0 -NA 2 -NP 1 tutorial -N 3
./vss-field-party.x 1 -ND 0 -NA 2 -NP 1 tutorial -N 3
./vss-field-party.x 2 -ND 0 -NA 2 -NP 1 tutorial -N 3
```

 或使用脚本

```
chmod +x Scripts/vss-field.sh
Scripts/vss-field.sh tutorial
```



