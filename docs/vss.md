## 基于向量空间秘密共享的安全计算协议使用

**九月份更新：**  优化了基于向量空间秘密分享做矩阵乘法时offline阶段生成向量三元组的效率，从而提高了基于向量空间秘密分享的矩阵乘法的整体效率。优化后的矩阵乘法效率是上一个版本的11倍左右

向量空间秘密共享（Vector Space Secret Shaing）是Song等人发表于ACM CCS‘22 的安全多方学习框架pMPL所使用的底层秘密共享技术。
在Garnet中向量空间秘密共享技术所对应的虚拟机是vss-party。vss-party基于MP-SPDZ原生的semi-party和hemi-party。vss-party实现了基于向量空间秘密共享的三方安全计算操作（目前只支持64位）。

### 基础设置

设置ssl

```
Scripts/setup-ssl.sh 3
```

### 编译tutorial程序

设置输入、编译mpc程序、设置环参数

```
echo 1 2 3 4 > Player-Data/Input-P0-0
echo 1 2 3 4 > Player-Data/Input-P1-0
echo 1 2 3 4 > Player-Data/Input-P2-0
./compile.py -R 64 tutorial
```

### 编译vss-party虚拟机

```
make -j 8 vss-party.x
```
  
### 运行vss-party虚拟机

 在三个终端分别运行

```
./vss-party.x 0 tutorial
./vss-party.x 1 tutorial
./vss-party.x 2 tutorial
```

 或使用脚本

```
chmod +x Scripts/vss.sh
Scripts/vss.sh tutorial
```