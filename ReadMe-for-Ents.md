# ReadMe for Ents

We   currently supports versions of Linux 2014 and above, as well as MacOS High Sierra and later versions.

## Build Environment

To run Ents, we first need to build the running environment. The steps is as follows:

(1) Download the code:

```
git clone https://github.com/FudanMPL/Garnet.git
```

(2) Move to Garnet/ and create a new file named "CONFIG.mine",  add the following code to "CONFIG.mine"

```
MOD = -DRING_SIZE=32
MY_CFLAGS = -DINSECURE
```

(3) Run the following code in the terminal and wait for it to complete

```
make clean
make -j 8 tldr
make -j 8 replicated-ring-party.x
make -j 8 Fake-Offline.x
```

(4) Install pandas with pip

```
pip3 install pandas
```



## Data Preparation

 (1)Create a new dictionary named "Data" in Garnet/, and download the datasets into "Data". Here, we prepare the IRIS dataset for you to test, the download link is https://drive.google.com/drive/folders/1dLUA7VRHGNkvpH7cgIPIsRqLvO4nABb8?usp=sharing. The IRIS dataset includes three files: IRIS_test.csv, IRIS_train.csv and IRIS-efficiency.csv.

Note: for using other dataset, you need to also prepare the three csv files, and put them into the Data dictionary. Besides, you may need to modify the Garnet/Programs/Source/ents_accuracy.mpc and Garnet/Programs/Source/ents_efficiency.mpc to add the dataset information.

(2) Run the following code to generate ssl key

```
./Scripts/setup-ssl.sh 3
```

(3) Run the following code to generate offline data

```
./Scripts/setup-online.sh 3 32
```



## Running



To verify the efficiency of Ents, run the following code

```
python3 ./Scripts/efficiency_data_for_ents.py IRIS
python3 ./compile.py -R 32 ents_efficiency IRIS
./Scripts/ring-only-online.sh ents_efficiency-IRIS
```

To verify the accuracy of Ents, run the following code

```
python3 ./Scripts/accuracy_data_for_ents.py IRIS
python3 ./compile.py -R 32 ents_accuracy IRIS
./Scripts/ring-only-online.sh ents_accuracy-IRIS
```



