# 1 GFA默认优化

在编译时加入-A或--gfa选项，将自动优化sigmoid等常用非线性函数。
```
./compile.py -R 128 -A test_sigmoid
```

以rss为例，测试sigmoid优化情况。
```
./Scripts/setup-ssl.sh 3
make -j 8 tldr
make -j 8 replicated-ring-party.x
./Scripts/ring.sh test_sigmoid
```
测试结果如下。在准确度误差有限的情况下，轮次减少到了34
```
Trying to run 128-bit computation
expected 0.119, got 0.119202
expected 0.269, got 0.268889
expected 0.5, got 0.5
expected 0.731, got 0.731111
expected 0.88, got 0.880866
Significant amount of unused dabits of replicated Z2^128. For more accurate benchmarks, consider reducing the batch size with -b.
The following benchmarks are including preprocessing (offline phase).
Time = 0.0365383 seconds 
Data sent = 0.717456 MB in ~34 rounds (party 0)
Global data sent = 2.15237 MB (all parties)
```
# 2 自定义非线性函数GFA优化

在GFA模块中，生成所需函数的多项式，以非线性激活函数sigmoid为例

首先进入GFA目录
```
cd Compiler/GFA
```

创建sigmoid.py，在其中配置NFGen的相关参数

定义原始非线性函数的表达式
```
# fundenmental functions, indicating they are cipher-text non-linear operations.
def func_reciprocal(x):
        return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

# target function.
def sigmoid(x):
    return 1 * func_reciprocal((1 + func_exp(-x)))
```

设置函数名、定义域、定点数位数n和f等参数，其他可保持默认
```
# define NFD
sigmoid_config = {
    "function": sigmoid, # function config.
    "range": (-10, 10),
    "k_max": 10, # set the maximum order.
    "tol": 1e-3, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./sigmoig_spdz.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    # "test_graph": "./graph/" # (optional, need mkdir for this folder first), whether generate the graph showing the approximation and the real function.
}
```

运行该函数配置，将基于NFGen初始化多项式数据
```
python3 sigmoid.py
```
若控制台输出如下，说明函数近似多项式创建成功
```
>>>>> FINAL KM:  (6, 8)
```

在Programs/Source中编写mpc代码，导入NLA模块
```
from non_linear_app import appFunc
```

设置定点数精度，整数位和小数位位数。
```
sfix.set_precision(32, 64)
```
从appFunc库中获取函数f

使用f.At(x)即可计算f(x)
```
mysigmoid = appFunc('sigmoid')
for i in range(0, 5):
    test(mysigmoid.At(sfix(i-2)) ,a[i])
```
以rss虚拟机运行test_sigmoid.mpc进行测试
```
./Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
./compile.py -R 128 test_sigmoid
./Scripts/ring.sh test_sigmoid
```
终端输出如下
```
Trying to run 128-bit computation
expected 0.119, got 0.119173
expected 0.269, got 0.268871
expected 0.5, got 0.5
expected 0.731, got 0.731129
expected 0.88, got 0.880899
```