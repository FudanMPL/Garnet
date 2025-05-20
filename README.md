# Secure KNN Execution Process

requirement: Ubuntu 20.04

## 1. Deployment Process
```sh
cd Kona
make -j8 tldr

```

## 2. Offline Data Generate
```sh
make -j 8 knn-party-offline.x  # Compile the knn-party-offline.cpp file
./knn-party-offline.x  # Generate the required offline data

./Scripts/setup-ssl.sh 2
```

## 3. Run in the WAN setting

Configuring WAN Environment
```sh
./WAN.sh
```
Run the four experimental code configuration files in the WAN environment:
```sh
./KNN-experiment.sh
```
The experimental results are stored in the corresponding `KNN-experiment-res-WAN` folder.

## 4. Run in the LAN setting

Configuring LAN Environment
```sh
./WAN.sh
```
Run the four experimental code configuration files in the LAN environment:
```sh
./KNN-experiment.sh
```
The experimental results are stored in the corresponding `KNN-experiment-res-LAN` folder.



# Code Position

The codes of Kona, SecKNN, SecKNN-Triples, and SecKNN-DQBubble mainly locate at "Machines/kona.cpp", "Machines/secknn.cpp", "Machines/optimized-esd.cpp", and "Machines/optimized-top1.cpp", respectively.

# Note

Due to file size limit, we only provide one dataset (arcene). To perform more evaluation, you need to download dataset from [UCI repo](https://archive.ics.uci.edu/), put the dataset on the directory "Player-Data/Knn-Data/knn-1/", and modify  "Machines/kona.cpp", "Machines/secknn.cpp", "Machines/optimized-esd.cpp", and "Machines/optimized-top1.cpp" to add the dataset.
