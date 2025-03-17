#!/bin/bash

make clean


# 第一步：编译文件
echo "开始编译文件secknn ..."
make -j8  secknn.x > /dev/null 2>&1  # 假设使用 make 编译项目，若使用其他编译命令请替换

if [ $? -ne 0 ]; then
  echo "编译失败，退出脚本。"
  exit 1
fi

echo "编译完成！"

# 第二步：执行编译后的文件
echo "开始执行编译后的文件..."
nohup ./secknn.x 0 -pn 10000 -h localhost >> ./KNN-experiment-res-WAN/outputP0_secknn111.log 2>&1 &
nohup ./secknn.x 1 -pn 10000 -h localhost >> ./KNN-experiment-res-WAN/outputP1_secknn111.log 2>&1 &

# 等待所有后台任务完成
wait

echo "SecKNN的所有进程执行完毕!"

# 第一步：编译文件
echo "开始编译文件 kona..."
make -j8  kona.x > /dev/null 2>&1  # 假设使用 make 编译项目，若使用其他编译命令请替换

if [ $? -ne 0 ]; then
  echo "编译失败，退出脚本。"
  exit 1
fi

echo "编译完成！"

# 第二步：执行编译后的文件
echo "开始执行编译后的文件..."
nohup ./kona.x 0 -pn 10000 -h localhost >> ./KNN-experiment-res-WAN/outputP0_kona000.log 2>&1 &
nohup ./kona.x 1 -pn 10000 -h localhost >> ./KNN-experiment-res-WAN/outputP1_kona000.log 2>&1 &

# 等待所有后台任务完成
wait

echo "kona 的所有进程执行完毕!"




echo "程序执行完毕!"