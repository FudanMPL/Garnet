# 使用PSI协议

**12月份更新**：我们实现了一个基于OPRF的两方PSI协议，可以在两方虚拟机semi2k-party.x中使用。

### 功能介绍
使用我们的PSI功能可以获得两个参与方的Set数据集的交集，并且不暴露额外的信息。
![PSI for two parties](figs/psi.svg)

为了适配纵向隐私保护机器学习场景，我们还提供了基于PSI的两个参与方的纵向数据集对齐，对齐后的数据集可以用于隐私保护机器学习。具体功能为：两个参与方各自持有的数据集具有相同的ID空间但是是不同的特征（或标签）空间，首先获得两个参与方的ID数据的交集，并将ID交集内的来自两个参与方的所有特征值合并为一个数据集。

![align data from two party based on PSI](figs/psi-align.svg)


### API
默认双方的数据量是一样的，如果不一样，则先用假数据将小数据量的填充。
```python
# n:  number of data (maxmium)
# f0: number of types of features (including label) from P0
# f1: number of types of features (including label) from P1
# val: dataset after aligning features from two parties
# num: number of ID intersection
val,num = PSI(n,f0,f1)
```
### 使用
Scripts/PSI-Test/run_psi.sh是编译运行的脚本
```shell
bash Scripts/PSI-Test/run_psi.sh r # compile test_psi.mpc
bash Scripts/PSI-Test/run_psi.sh x # compile semi2k virtual machine
bash Scripts/PSI-Test/run_psi.sh t # run test_psi program in two-party
```