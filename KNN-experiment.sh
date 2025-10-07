#!/bin/bash

make clean


mkdir -p KNN-experiment-res

# Step 1: Compile files
echo "Starting to compile secknn ..."
make -j8  secknn.x > /dev/null 2>&1  # Use make to compile the project, replace if using other compilation commands

if [ $? -ne 0 ]; then
  echo "Compilation failed, exiting script."
  exit 1
fi

echo "Compilation complete!"

# Step 2: Execute compiled files
echo "Starting to execute compiled files..."
nohup ./secknn.x 0 -pn 10000 -h localhost > ./KNN-experiment-res/outputP0_secknn.log 2>&1 &
nohup ./secknn.x 1 -pn 10000 -h localhost > ./KNN-experiment-res/outputP1_secknn.log 2>&1 &

# Wait for all background tasks to complete
wait

echo "All SecKNN processes completed!"

# Step 1: Compile files
echo "Starting to compile kona..."
make -j8  kona.x > /dev/null 2>&1  # Use make to compile the project, replace if using other compilation commands

if [ $? -ne 0 ]; then
  echo "Compilation failed, exiting script."
  exit 1
fi

echo "Compilation complete!"

# Step 2: Execute compiled files
echo "Starting to execute compiled files..."
nohup ./kona.x 0 -pn 10000 -h localhost > ./KNN-experiment-res/outputP0_kona.log 2>&1 &
nohup ./kona.x 1 -pn 10000 -h localhost > ./KNN-experiment-res/outputP1_kona.log 2>&1 &

# Wait for all background tasks to complete
wait

echo "All kona processes completed!"




# Step 1: Compile files
echo "Starting to compile optimized's esd ..."
make -j8  optimized-esd.x > /dev/null 2>&1  # Use make to compile the project, replace if using other compilation commands

if [ $? -ne 0 ]; then
  echo "Compilation failed, exiting script."
  exit 1
fi

echo "Compilation complete!"

# Step 2: Execute compiled files
echo "Starting to execute compiled files..."
nohup ./optimized-esd.x 0 -pn 10000 -h localhost > ./KNN-experiment-res/outputP0_optimized-esd.log 2>&1 &
nohup ./optimized-esd.x 1 -pn 10000 -h localhost > ./KNN-experiment-res/outputP1_optimized-esd.log 2>&1 &

# Wait for all background tasks to complete
wait

echo "All optimized-esd processes completed!"



# Step 1: Compile files
echo "Starting to compile optimized's top1 ..."
make -j8  optimized-top1.x > /dev/null 2>&1  # Use make to compile the project, replace if using other compilation commands

if [ $? -ne 0 ]; then
  echo "Compilation failed, exiting script."
  exit 1
fi

echo "Compilation complete!"

# Step 2: Execute compiled files
echo "Starting to execute compiled files..."
nohup ./optimized-top1.x 0 -pn 10000 -h localhost > ./KNN-experiment-res/outputP0_optimized-top1.log 2>&1 &
nohup ./optimized-top1.x 1 -pn 10000 -h localhost > ./KNN-experiment-res/outputP1_optimized-top1.log 2>&1 &

# Wait for all background tasks to complete
wait

echo "All optimized-top1 processes completed!"








echo "Program execution completed!"