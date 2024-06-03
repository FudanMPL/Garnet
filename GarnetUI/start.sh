#!/bin/bash

# 激活虚拟环境
source /root/miniconda3/bin/activate py310
# 进入目录并执行命令
cd Garnet/GarnetUI
chmod +x env.sh
bash env.sh
cd Vue3-garnet-ui
pnpm dev --host 0.0.0.0 --port 9000 &
cd ../
python3 ui/manage.py runserver 0.0.0.0:8000 &
python3 ui/manage.py qcluster &
# 保持容器运行
while true; do
    sleep 10
done
