## 基于NFGen的非线性函数近似计算

清华团队Xiaoyu Fan等人发表在CCS'2022上的论文NFGen中，提出了NFGen工具包。NFGen利用离散分段多项式，自动化综合考虑后端MPC协议的开销、近似精度等指标，生成较优的近似多项式，再利用后端的MPC协议库进行计算并生成对应的代码模板。在Garnet中基于NfGen的多项式生成模块实现了GFA，general nonlinear-function approximation，通用非线性函数近似计算模块，能支持复杂非线性函数的mpc友好的计算，并可结合Function Secret Sharing进一步提高效率。

### 1 Garnet预设非线性函数

从预设函数库中导入所需函数如sigmoid，即直接带有近似计算

```
from GFA.presets import sigmoid
```
预设函数库presets中包含了常用的机器学习激活函数和标准正态分布函数如下
| 预设函数    | 表达式                                                       | 阶数 | 精度（n, f） | 段数 | 范围        |
| ----------- | ------------------------------------------------------------ | ---- | ------------ | ---- | ----------- |
| sigmoid     | 1 / (1 + math.exp(-x))                                       | 6    | （44，96）   | 8    | （-10，10） |
| tanh        | math.tanh(x)                                                 | 5    | （44，96）   | 10   | （-50，50） |
| soft_plus   | math.log(1 + math.exp(x))                                    | 8    | （44，96）   | 9    | （-20，50） |
| elu         | [ELU — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html) | 4    | （44，96）   | 8    | （-50，20） |
| selu        | [SELU — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html) | 4    | （44，96）   | 9    | （-50，20） |
| gelu        | [GELU — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) | 4    | （44，96）   | 10   | （-20，20） |
| snormal_dis | (1 / math.sqrt(2 * math.pi)) * math.exp(-x ** 2 / 2)         | 8    | （44，96）   | 13   | （-10，10） |


设置对应定点数位数后可计算该函数的近似结果
```
from GFA.presets import sigmoid
sfix.set_precision(44, 96)
x = sfix(0)
y = sigmoid(x)
```

### 2 自定义函数近似优化

使用GFA模块可生成自定函数的近似计算多项式，在Source/test_gfa.mpc中以自定义的sigmoid函数为例

1. 在Programs/Source中在新建mpc代码，导入GFA模块并在计算前设置对应定点数位数

```
from gfapp import GFA
sfix.set_precision(31, 63)
```
2. mpc代码中创建原始非线性函数的表达式，在目标函数前加上GFA装饰器。在装饰器中可修改GFA参数，多项式最大阶数k、定点数小数位数f和位数n、近似范围、导数是否存在等。（不设置将默认为10，44，96，（-10，10）, True）
```
# target function.
@GFA(10, 31, 63, (-10,10))
def mysigmoid(x):
    return 1 / ((1 + sp.exp(-x)))
```

3. 在后续代码中即可直接使用该函数计算，将自动具有近似优化
```
print_ln('using GFA sigmoid')
for i in range(0, 5):
    expected = sigmoid(sfix(i-2))
    actual = mysigmoid(sfix(i-2))
```

以rss虚拟机运行test_gfa.mpc进行测试
```
./Scripts/setup-ssl.sh 3
make -j 8 tldr
make -j 8 replicated-ring-party.x
./compile.py -R 128 test_gfa
./Scripts/ring.sh test_gfa
```
输出如下，计算结果与Garnet精确计算误差有限
```
Trying to run 128-bit computation
using GFA sigmoid
expected 0.119203, got 0.119202
expected 0.268941, got 0.268889
expected 0.5, got 0.5
expected 0.731059, got 0.731111
Significant amount of unused dabits of replicated Z2^128. For more accurate benchmarks, consider reducing the batch size with -b.
The following benchmarks are including preprocessing (offline phase).
Time = 0.0581284 seconds 
Data sent = 1.14333 MB in ~138 rounds (party 0)
Global data sent = 3.42998 MB (all parties)
```