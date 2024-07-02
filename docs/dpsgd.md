# PEA论文中的安全DPSGD算法

《Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy》是Wenqiang Ruan等人发表在IEEE S&P 2023上的文章，提出了一个安全差分隐私随机梯度下降（DPSGD）协议。差分隐私随机梯度下降算法中的两个关键操作为通过计算平方根倒数对梯度的L2范数进行裁剪、向裁剪后的梯度添加随机高斯噪声。

## 数据准备

首先需要生成用于高斯加噪步骤的随机数。
在Garnet文件夹下运行如下命令。
```
cd Utils/gauss_test
g++ generate_noise.cpp -o generate_noise
./generate_noise <number_of_files> <numbers_per_file>
```
其中"number of files","numbers_per_file"分别代表参与方的数量以及一个不小于模型参数量的值。例如，
```
./generate_noise 2 1000
```

## 模型构建

以一个简单的逻辑回归模型为例。在Programs/Source文件夹下新建一个mpc文件，命名为autograd_logistic_dpsgd.mpc，并将如下测试代码复制到该文件中。
```
import nn as nn
from tensor import Tensor,autograd_function
import optimizer as optim
import dataloader as dataloader
import functional as F
# program.use_trunc_pr = True

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, dp = True)

    def forward(self,x):
        out = self.linear(x)
        return out

x = MultiArray([5, 3], sfix)
y = MultiArray([5, 10], sfix)

@for_range(x.total_size())
def _(i):
    x.assign_vector(sfix(i/10), i)
y.assign_all(0)
for i in range(5):
    y[i][i//3] = 1

dataloader = dataloader.DataLoader(x, y, batch_size = 3)
model = LogisticRegression(3, 10)
optimizer = optim.DPSGD(model.parameters(), lr = 0.01, iterations = 1)
criterion = nn.MSELoss(reduction = 'none')
model.train()
@for_range(1)
def _(i):
    x, labels = dataloader[i]
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, labels)
    loss.print_reveal_nested()
    print(loss.sizes)
    loss.backward()
    optimizer.step()
```

## 参数调整

与普通的SGD优化器不同，DPSGD拥有不同的梯度结构，因此需要传入额外的参数用以区分。
首先，在线性层的构建中，需要修改参数dp的值，如下。
```
self.linear = nn.Linear(n_inputs, n_outputs, dp = True)
```

其次，优化器选择DPSGD优化器并传入额外的迭代次数参数，如下。
```
optimizer = optim.DPSGD(model.parameters(), lr = 0.01, iterations = 1)
```

最后，均方误差损失函数初始化中传入防止梯度聚合的参数，如下。
```
criterion = nn.MSELoss(reduction = 'none')
```

## 模型运行

在Garnet文件夹下运行如下命令。
```
make -j 8 sml-party.x
./compile.py -l -R 64 autograd_logistic_dpsgd
Scripts/sml.sh autograd_logistic_dpsgd
```