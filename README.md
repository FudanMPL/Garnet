# README for Ents


This README file provides instructions for reproducing the experimental results in the paper "Ents: An Efficient Three-party Training Framework for Decision Trees by Communication Optimization" (CCS 2024).


**RoadMap:**
You should first download the source code according to [Download Source Code](#download).
Then, you should set up the testing environment according to [Set Up Environment](#environment).
Finally, you can evalute the efficiency or accuracy of the frameworks according to [Efficiency Evaluation](#efficiency) and [Accuracy Evaluation](#accuracy), respectively.


## Download Source Code
<a name="download"></a>


You can download the source code by 

```
git clone https://github.com/FudanMPL/Garnet.git -b Ents
```

or download the .zip file.


## Set Up Environment
<a name="environment"></a>


We provide a dockerfile to set up the testing environment.

### 1. Build Docker Image

Use the following command to build the Docker image:

```
sudo docker build -t ents .
```

Building the image may take some time.

Ignore the error info if the images are finally build successfully.

### 2. Launch Docker Container 

Launch the Docker container and access its shell using:

```
sudo docker run --cap-add=NET_ADMIN -it ents bash
```



## Efficiency Evaluation
<a name="efficiency"></a>


### 0. Network Simulation


Before evaluating the efficiency of the frameworks, you should simulate the network setting.

If you want to evaluate the efficiency of the frameworks in the LAN setting, use the following command
```
./lan.sh
```

If you want to evaluate the efficiency of the frameworks in the WAN setting, use the following command 
```
./wan.sh
```

Ignore the following error if it appears:

```
Error: Cannot delete qdisc with handle of zero.
```


### 1. Main Experiment (Table 3 in Section 5.2)

#### (1) Ents

To evaluate the efficiency of Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_efficiency iris
./Scripts/ring.sh -F ents_efficiency-iris
```

It will output as follows:

```
Using security parameter 40
Trying to run 32-bit computation
ents dataset=iris h=6
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
training 4-th layer
training 5-th layer
training 6-th layer (leaf layer)
REWINDING - ONLY FOR BENCHMARKING
The following benchmarks are excluding preprocessing (offline phase).
Time = 1.17903 seconds 
Data sent = 9.46039 MB in ~15931 rounds (party 0; rounds counted double due to multi-threading)
Global data sent = 34.1352 MB (all parties)
```

The runtime, communciation rounds and communication size are  1.17903s, 15931 and 34.1352 MB respectively.

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.



#### (2) Hamada et al.'s Framework

To evaluate the efficiency of Hamada et al.'s framework on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada iris
./Scripts/ring.sh -F hamada-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (3) Abspoel et al.'s Framework

To evaluate the efficiency of Abspoel et al.'s framework on the 'iris' dataset, execute these commands sequentially:


```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 abspoel iris
./Scripts/ring.sh -F abspoel-iris
```


You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.

Note: Compiling for 'adult' and 'skin-segmentation' (i.e. runing the follow commands) may take 1-2 days.
```
python3 ./compile.py -R 128 abspoel adult (or skin-segmentation)
```


### 2. Ablation Experiment (Figure 2 in Section 5.3)

#### (1) Ents
To evaluate the efficiency of Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_efficiency iris
./Scripts/ring.sh -F ents_efficiency-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (2) Hamada et al.'s Framework
To evaluate the efficiency of Hamada et al.'s framework on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada iris
./Scripts/ring.sh -F hamada-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (3) Hamada et al.'s Framework-radixsort

To evaluate the efficiency of Hamada et al.'s framework-radixsort on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada_radixsort iris
./Scripts/ring.sh -F hamada_radixsort-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (4) Hamada et al.'s Framework-extend

To evaluate the efficiency of Hamada et al.'s framework-extend on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 hamada_extend iris
./Scripts/ring.sh -F hamada_extend-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


### 3. Share Conversion Protocol (Table 4 in Section 5.4)

#### (1) ConvertShare

To evaluate the efficiency of the protocol ConvertShare with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion 10
./Scripts/ring.sh -F test-conversion-10
```
 


#### (2) Convert-A2B

To evaluate the efficiency of the protocol Convert-A2B with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion-with-a2b 10
./Scripts/ring.sh -F test-conversion-with-a2b-10
```

#### (3) Convert-Dabits

To evaluate the efficiency of the protocol Convert-Dabits with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion-with-dabits 10
./Scripts/ring.sh -F test-conversion-with-dabits-10
```

#### (4) Convert-FSS

To evaluate the efficiency of the protocol Convert-FSS with vector of size n (such as 10), execute these commands sequentially:

```
./Fake-Offline.x 3 -e 15,31,63
make -j 6 fss-ring-party.x
python3 ./compile.py -R 128 test-conversion-with-fss 10
./Scripts/fss-ring.sh -F test-conversion-with-fss-10
```



### 4. Efficiency of Two-Party Ents (Table 6 in Appendix E.2) 
To  evaluate the efficiency of two-party Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_two_party iris
./Scripts/semi2k.sh -F ents_two_party-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.



## Accuracy Evaluation (Table 7 in Appendix F)
<a name="accuracy"></a>


### 1. Ents

To evaluate the the accuracy of Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_accuracy.py iris
python3 ./compile.py -R 32 ents_accuracy iris
./Scripts/ring.sh -F ents_accuracy-iris
```

It will output as follows

```
Using security parameter 40
Trying to run 32-bit computation
training 0-th layer
training 1-th layer
training 2-th layer
training 3-th layer
training 4-th layer
training 5-th layer
training 6-th layer (leaf layer)
iris-test for height 6: 48/51
```

The accuracy is 48/51 = 0.9411.


**Note:** The accuracy may vary with each run and might differ from the result in Table 7. That is because when multiple best split points, Ents will randomly select one.
 
You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


### 2. scikit-learn

To evaluate the the accuracy of scikit-learn on the 'iris' dataset, execute the following command:


```
python3 ./Scripts/ski-dt.py iris
```

It will output as follows

```
scikit-learn accuracy on iris: 0.9411764705882353
```

**Note:** The accuracy may vary with each run and might differ from the result in Table 7. That is because when multiple best split points, scikit-learn will randomly select one.
 
You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


 