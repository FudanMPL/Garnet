# 1 使用预设函数的近似计算

从预设函数库中导入所需函数如sigmoid，即直接带有近似计算
```
from GFA.presets import sigmoid
```

以rss为例，测试sigmoid优化情况。
```
./Scripts/setup-ssl.sh 3
make -j 8 tldr
make -j 8 replicated-ring-party.x
./compile.py -R 128 test_sigmoid
./Scripts/ring.sh test_sigmoid
```
测试结果如下。
```
Trying to run 128-bit computation
expected 0.119, got 0.119202
expected 0.269, got 0.268889
expected 0.5, got 0.5
expected 0.731, got 0.731111
expected 0.88, got 0.880866
Significant amount of unused dabits of replicated Z2^128. For more accurate benchmarks, consider reducing the batch size with -b.
The following benchmarks are including preprocessing (offline phase).
Time = 0.0389078 seconds 
Data sent = 0.717456 MB in ~34 rounds (party 0)
Global data sent = 2.15237 MB (all parties)
```
# 2 使用自定义函数近似优化

在GFA模块中，生成所需函数的多项式，以非线性激活函数sigmoid为例

1. 在Programs/Source中在新建mpc代码，导入GFA模块并在计算前设置好定点数精度n和f

```
from gfapp import GFA
sfix.set_precision(31, 63)
```
2. mpc代码中创建原始非线性函数的表达式，并在最终的目标函数前加上GFA装饰器。在装饰器中可修改GFA参数，最大段数k、定点数位数n和f、近似范围等。（不设置将默认为10，31，63，（-10，10））
```
# fundenmental functions, indicating they are cipher-text non-linear operations.
def func_reciprocal(x):
        return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

# target function.
@GFA(10, 31, 63, (-10,10))
def mysigmoid(x):
    return 1 * func_reciprocal((1 + func_exp(-x)))
```

3. 在后续代码中即可直接使用原函数计算，将自动具有GFA优化
```
print_ln('using GFA sigmoid')
for i in range(0, 5):
    test(mysigmoid(sfix(i-2)) ,a[i])
```

以rss虚拟机运行test_gfa.mpc进行测试
```
./Scripts/setup-ssl.sh 3
make -j 8 tldr
make -j 8 replicated-ring-party.x
./compile.py -R 128 test_gfa
./Scripts/ring.sh test_gfa
```
终端输出如下，计算结果与迭代法所获得的精度相近
```
Using security parameter 40
Trying to run 128-bit computation
using original sigmoid
expected 0.119, got 0.119203
expected 0.269, got 0.268941
expected 0.5, got 0.5
expected 0.731, got 0.731059
expected 0.88, got 0.880797
using GFA sigmoid
expected 0.119, got 0.119202
expected 0.269, got 0.268889
expected 0.5, got 0.5
expected 0.731, got 0.731111
expected 0.88, got 0.880866
Significant amount of unused dabits of replicated Z2^128. For more accurate benchmarks, consider reducing the batch size with -b.
The following benchmarks are including preprocessing (offline phase).
Time = 0.07801 seconds 
Data sent = 1.34877 MB in ~139 rounds (party 0)
Global data sent = 4.0463 MB (all parties)
This program might benefit from some protocol options.
Consider adding the following at the beginning of 'test_gfa.mpc':
        program.use_trunc_pr = True
        program.use_split(3)
```