# mPSI协议

**2025年3月份更新**：三方PSI效率优化，实现了一个基于OPRF和OKVS的三方PSI协议。

## 运行基于OPRF和OKVS的三方PSI协议

### 功能介绍
使用我们的PSI功能可以获得三个参与方的Set数据集的交集，并且不暴露额外的信息。

### API
默认双方的数据量是一样的，如果不一样，则先用假数据将小数据量的填充。
第i方的输入数据存储在"Player-Data/PSI/ID-Pi"
输出的交集结果存储在"Player-Data/PSI/ID-InterSet"
```python
# n:  number of parties
mPSI(n)
```

### 使用
Scripts/PSI-Test/run_psi.sh是编译运行的脚本
默认运行的mpc文件名是：test_mpsi.mpc，如果想运行其他文件可修改脚本run_psi.sh
```shell
bash Scripts/PSI-Test/run_mpsi.sh c # compile test_psi.mpc
bash Scripts/PSI-Test/run_mpsi.sh x # compile virtual machine replicated-ring-party.x
bash Scripts/PSI-Test/run_mpsi.sh i # run test_psi program in rss, i \in {0,1,2}
```