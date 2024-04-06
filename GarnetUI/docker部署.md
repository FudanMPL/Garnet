# docker 部署步骤

# 拉取ubuntu22.04镜像
```bash
docker pull ubuntu:22.04
```
# 创建并运行容器 
```bash
docker run -it --name garnetui-3.7 ubuntu:22.04
```

# 安装conda 并通过conda创建虚拟环境 安装python310
```bash
docker cp Miniconda3-latest-Linux-x86_64.sh garnetui-3.6:/
```

# 启动并进入一个已经停止运行的容器
```bash
docker start garnetui-3.8
docker attach garnetui-3.8
```
# 安装conda
```bash
bash Miniconda3-latest-Linux-x86_64.sh
conda --version
```

# 安装py310 并进入310的环境
```bash
conda create -n py310 python=3.10
conda activate py310
python --version
```

# 安装npm 10.2 和node 21.2
```bash
apt update
apt install nodejs
apt install npm
```

# 更新node版本
```bash
npm install -g n
apt install wget
n 21.2
hash -r
```

# 安装git
```bash
apt install git
```

# clone garnet
```bash
git clone --depth 1 https://github.com/FudanMPL/Garnet.git
```

# 外部库准备 编译
```bash
cd Garnet
apt-get install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4  texinfo yasm
# 可能要开代理
make -j 8 tldr
```

# clash for linux
```bash
git clone https://github.com/Elegycloud/clash-for-linux-backup.git
cd clash-for-linux-backup
apt install vim
chmod +rw .env

vim .env # 修改clash url
apt install curl
bash start.sh
source /etc/profile.d/clash.sh
proxy_on # proxy_off
```

# 安装前后端依赖
```bash
cd GarnetUI
pip install --default-timeout=300 -r piplist
cd Vue3-garnet-ui
npm install -g pnpm # 需要全局部署
npm install
```

# 配置env.sh 并启动
```bash
chmod +x env.sh
```

# 将容器封装为镜像
```bash
docker commit my_container my_image:v1.0
```

# 配置端口映射
```bash
docker run --name garnet-3.11 -p  10.176.34.173:8000:80 -p 10.176.34.173:9000:90 -it garnet:v1.1
```

# 使用后台命令启动前后端和django q 229 276 279
```bash

```

# 安装boost1.75
```bash
# 清除旧版本
rm -rf /usr/include/boost
# 安装新版本
wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
# 解压缩
tar -xzvf boost_1_75_0.tar.gz

# 配置
cd boost_1_75_0
./bootstrap.sh --with-libraries=all
./b2
./b2 install

# 检查版本
cat /usr/include/boost/version.hpp | grep "BOOST_LIB_VERSION"

```

# 编译虚拟机
make -j 8 semi2k-party.x

# 手动创建Input Output 文件夹

# 启动前后端 和 django q
```bash
cd Vue3-garnet-ui
nohup pnpm dev --host 0.0.0.0 --port 9000 &

nohup python3 ui/manage.py qcluster &

nohup python3 ui/manage.py runserver 0.0.0.0:8000 &
```


docker host 方式

# 封装为镜像
需要先清理数据库
需要配置环境变量
需要启动前后端和django q 任务


```bash
docker rm garnet173  
docker rmi garnet173-3.14:latest 
docker build -t garnet173-3.14 . 
docker run -it --network=host --name garnet173 garnet173-3.14:latest
```