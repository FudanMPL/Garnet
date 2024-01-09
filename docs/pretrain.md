## 运行预训练模型安全微调

### 环境配置

在主目录（/Garnet）下提供了requirements.txt和fine-tuning.yaml用于配置环境，可以在主目录下执行以下命令完成环境配置

```
pip install -r ./requirements.txt 
```

### 数据准备
生成的模型和数据可以从https://drive.google.com/file/d/1OOBGQroO4YmlBAPqEbz1VUHNiKBTDXGB/view?usp=sharing 得到。其中数据来自于[1] , [2]和[3]。下载后解压将三个文件夹及其内容拷贝至/Garnet/Compiler/DL目录下即可


### 代码运行
  以LeNet和[CK+48][1]数据集为例，若只希望载入自己的预训练模型进行安全fine-tuning，只需依次执行下列脚本即可：


1. 获取预训练模型，在Garnet主目录运行如下命令
```
cd Compiler/DL
python LeNet-Ferplus.py
```
2. 获取适用于Garnet的训练数据
```
python ./CK+48-data-full.py
```
3. 编译安全fine-tuning的mpc文件
```
cd ../../
python compile.py -R 64 torch_lenet_fine-tuning
```
4. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
5. 在虚拟机中运行编译好的文件，进行fine-tuning
```
Scripts/ring.sh torch_lenet_fine-tuning  
```

  若希望在Garnet中使用带有安全模型选择协议的安全fine-tuning，则可以通过执行下列脚本完成，在这里我们展示从两个预训练模型权重中选取最适合于CK+48数据集fine-tuning的权重的例子，权重对应的数据集分别为[FER+][2]和[CIFAR100][3]：


1. 获取一批预训练模型,，在Garnet主目录运行如下命令
```
cd ./Compiler/DL
python LeNet-Ferplus.py
python LeNet-CIFAR100.py
```
2. 获取适用于Garnet的训练数据
```
python ./CK+48-data-full.py
```
3. 获取预训练数据与用于fine-tuning的数据的平均特征
```
python ./GetAll-feature.py
```
4. 编译带有安全模型选择协议的安全fine-tuning的mpc文件
```
cd ../../
python compile.py -R 64 torch_ckplus48_lenet_selected
```
5. 创建证书和密钥并编译RSS虚拟机
```
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
```
6. 在虚拟机中运行编译好的文件，完成模型选择并进行fine-tuning
```
Scripts/ring.sh torch_ckplus48_lenet_selected
```


[1]: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/tree/master/CK+48
[2]: https://github.com/microsoft/FERPlus
[3]: http://www.cs.toronto.edu/~kriz/cifar.html
