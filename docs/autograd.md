## 自动微分模块
自动微分模块使得用户只需要按照PyTorch的语法接口定义模型即可构建机器学习模型的安全训练或推理过程，从而极大地提升Garnet用于完成安全多方学习任务时的易用性。该模块共包含六个子模块：安全矩阵计算模块、Tensor模块、Functional模块、NN模块、Optimizer模块、DataLoader模块，其中Tensor、Functional、NN、Optimizer、DataLoader模块可以直接被用户调用，用于构建机器学习模型的安全。


### 环境配置
在主目录（/Garnet）下提供了requirements.txt用于配置环境，可以在主目录下执行以下命令完成环境配置.

```
pip install -r ./requirements.txt 
```

### 模型构建
以逻辑回归模型为例。在Programs/Source文件夹下新建一个mpc文件，命名为autograd_logistic.mpc，并将如下代码复制到该文件中，即可完成一个逻辑回归模型训练过程的构建。
```
import nn as nn
from tensor import Tensor,autograd_function
import optimizer as optim
import dataloader as dataloader
import functional as F

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
    def forward(self,x):
        out = F.sigmoid(self.linear(x))
        return out

x = Tensor.ones(30, 10)
y = Tensor.ones(30, 2)
dataloader = dataloader.DataLoader(x, y, batch_size = 6)
model = LogisticRegression(10, 2)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()
model.train()
@for_range(5)
def _(i):
    x, labels = dataloader[i]
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, labels)
    loss.print_reveal_nested()
    loss.backward()
    optimizer.step()
```

### 模型运行
在Garnet文件夹下运行如下命令，在位长为64位的环上运行上述逻辑回归模型训练过程。
```
python compile.py -R 64 autograd_logistic
./Scripts/ring.sh autograd_logistic
```