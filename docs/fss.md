## 运行Function Secret Sharing与Replicated Secret Sharing混合协议

**九月份更新:** 我们针对Function Secret Sharing的通信进行了并行优化，使得Function Secret Sharing相比Replicated Secret Sharing在比较函数上的通信轮次减少了2轮（如果去除offline阶段的signal通信则能进一步降低至3轮）。实验数据显示在LAN下Function Secret Sharing相比原本提速了3倍。该优化极大地提高了Garnet在训练机器学习模型时的效率。


本协议通过Function Secret Sharing减小比较的通信量（通信轮次待优化），在乘法、加法运算时转换回Replicated Secret Sharing进行计算，兼容NFGen并提供分段时的加速。

### 环境配置
由于fss-ring-party.x默认会生成edabits，同时通信量开销会计算online+offline两个部分，因此我们需要开启online benchmark only，为此我们需要创建一个CONFIG.mine文件在Garnet目录下，并在文件中添加"MY_CFLAGS = -DINSECURE"。

接着在Garnet/Player-Data文件夹下创建2-fss文件夹，并编译fss-ring-party.x虚拟机
编译过程可能会花费几分钟时间
```
cd Garnet/Player-Data
mkdir 2-fss
cd ..
make clean
make -j 8 fss-ring-party.x
```

下一步，在控制台上输入以下命令，生成证书及密钥

```
./Scripts/setup-ssl.sh 3
```

最后，生成Offline需要的内容
```
make -j 8 Fake-Offline.x
./Fake-Offline.x 3 -e 15,31,63
```

### 编译mpc文件
想要开启function secret sharing做比较，需要在编译时开启cisc指令并指定LTZ指令不在编译器层进行拆解：

编译test_sfix.mpc获得编译后的二进制文件，该过程可能会花费几分钟时间
```
./compile.py -l -R 128 -C -K LTZ test_sfix
```

### 运行

运行test_sfix只需要执行./Scripts/fss-ring.sh -F test_sfix，其中-F表示开启online benchmark only，需要注意的是，如果输出结果中没有例如“c is 1 , a-b is -478.79”的内容，则表示判断大小的结果均是正确的，否则表示出现了错误情况。

```
./Scripts/fss-ring.sh -F test_sfix

以下为控制台输出
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 0 -F test_sfix -pn 11279 -h localhost
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 1 -F test_sfix -pn 11279 -h localhost
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 2 -F test_sfix -pn 11279 -h localhost
Using security parameter 40
Trying to run 128-bit computation
REWINDING - ONLY FOR BENCHMARKING
The following benchmarks are excluding preprocessing (offline phase).
Time = 69.982 seconds 
Data sent = 1.12 MB in ~70000 rounds (party 0)
Global data sent = 3.8 MB (all parties)
This program might benefit from some protocol options.
Consider adding the following at the beginning of 'test_sfix.mpc':
        program.use_split(3)

想要测试FSS究竟减少了多少通信量，只需要关闭FSS的比较，通过重新编译test_sfix.mpc文件，关闭-C -K LTZ选项即可，即，运行
./compile.py -l -R 128 test_sfix
接着运行test_sfix文件即可（关闭-C -K LTZ选项后，fss-ring-party.x与replicated-ring-party.x内容一致）
此时，控制台输出如下
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 0 -F test_sfix -pn 11531 -h localhost
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 1 -F test_sfix -pn 11531 -h localhost
Running /home/txy/Garnet/Scripts/../fss-ring-party.x 2 -F test_sfix -pn 11531 -h localhost
Using security parameter 40
Trying to run 128-bit computation
REWINDING - ONLY FOR BENCHMARKING
The following benchmarks are excluding preprocessing (offline phase).
Time = 7.42893 seconds 
Data sent = 9.92 MB in ~90000 rounds (party 0)
Global data sent = 29.76 MB (all parties)
This program might benefit from some protocol options.
Consider adding the following at the beginning of 'test_sfix.mpc':
```

通过上述结果可以看到通信量减少了～10倍，通信轮次的减少以及本地计算的加速将在后续版本陆续更新，敬请期待。
