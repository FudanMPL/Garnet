

# 字符串安全操作
我们在Garnet中增加了字符串处理功能，具体使用方法如下：
## 单字符处理
首先，基于utf-8编码，我们设计了表示单个字符类型数据的明文和密文状态下的两种数据类型。针对单个字符的schr和cchr两种数据类型，分别表示密文状态下的字符类型和明文状态下的字符类型，同时给出打印接口print_cchr()，使用示例如下：
```python
a=schr.get_input_from(0) #从参与方P0终端获取输入字符，并将字符转换成密文形式赋值给a
b=schr('人')#直接赋值操作，将字符'人'转换成密文形式赋值给b
c=cchr('之')#直接赋值操作，将字符'之'转换成明文形式赋值给c
print_cchr(b.reveal()) #打印出 人
print_cchr(c)#打印出 之

```
## 字符串处理
基于上述的处理单个字符串的schr和cchr两种数据类型，我们实现了sstring类，表示密文状态下的字符串数据类型，并在sstring类中实现了切片，比较，打印数据的操作。具体使用方法示例如下：
```python
s1=sstring('人之初，性本善，性相近，习相远')
s1[2]='C' #索引2位置进行赋值操作
s1.print_reveal_nested() #打印出 人之C，性本善，性相近，习相远
s2=sstring('to be or not to be！')
s2.print_reveal_nested() #打印出 to be or not to be！
s3=s1[:4] #切片操作,并将结果赋值给s3
s3.print_reveal_nested() #打印出 人之初，
```

## 场景示例：
在三方复制秘密共享协议下，假设P0，P1，P2三方手里分别有两个用户的姓名和身份证信息。我们需要对这些用户信息进行汇总并去掉重复的用户信息，最终将结果打印出来，代码示例如下：
### 三个参与方输入数据准备：
P0终端输入数据（文件位置：/Garnet/Player-Data/Input-P0-0):
```python
420101199707298148 陈欣子 
370101199302127797 周欣怡
```
P1终端输入数据（文件位置：/Garnet/Player-Data/Input-P1-0):
```python
620101196903199717 郑慧源
420101198608177610 张弛\x00
```

P2终端输入数据（文件位置：/Garnet/Player-Data/Input-P2-0):
```python
620101196903199717 郑慧源
420101198608177610 张弛\x00
```
### mpc代码：
在Garnet/Programs/Source/test_str.mpc路径下书写以下代码：
```python
User_ID=[] #存储身份证信息
for player in [0,1,2]:#三个参与方，P0，P1，P2
    for i in range(2):#每个参与方输入两个用户信息
        Not_exist=sint(0)#指示该用户信息是否已经存在
        ID=sstring(length=18) #存储用户身份证ID
        for i in range(18):
            ID[i]=schr.get_input_from(player)
        Name=sstring(length=3)#存储用户名字
        for i in range(3):
            Name[i]=schr.get_input_from(player)
            
        for itm in User_ID:#使用ID判断该用户是否已经存在
            Not_exist = Not_exist + (ID==itm).reveal()
        @if_(Not_exist.reveal()==0)
        def _():
            Name.print_reveal_nested(end='\t')
            ID.print_reveal_nested()
            User_ID.append(ID)
            
```
### 编译运行
在Garnet/路径下使用如下命令进行编译test_str.mpc文件，生成对应的字节码文件
```shell
./compile test_str
```
根据使用复制秘密共享协议的假设，我们使用replicated-ring-party.x虚拟机，运行以下两行代码：
```shell
make -j 6 replicated-ring-party.x
./Scripts/ring.sh test_str
```
### 运行结果：
```python
陈欣子  42010119970729814X
周欣怡  370101199302127797
郑慧源  620101196903199717
张弛    420101198608177610
```
**注意：**

1. 我们实现了转义字符的读入和处理方案，比如\x00就代表ascii码为0的字符，\xff代码ascii码为十六进制ff的字符，\x00为空字符。同时，我们实现了如下特殊的转义字符：\t（水平制表符），\n（换行符），\\ （\）。
2. 字符串类型必须输入固定长度的数据，当实际字符串长度小于所设定的长度时，可以用转义字符\x00填充到指定长度。例如，上面的示例场景中，用于接收用户输入姓名的sstring类的长度被指定为3，但是对于张弛这个名字只有2个字符，因此我们在该名字后面使用\x00进行填充到长度为3。















