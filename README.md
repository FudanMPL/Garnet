# README for Ents


This README file provides instructions for reproducing the experimental results in the paper "Ents: An Efficient Three-party Training Framework for Decision Trees by Communication Optimization" (CCS 2024).

We thank all anonymous reviewers for their insightful comments, which have significantly contributed to the improvement of this artifact..


**System Requirements:** 
Unless specified differently, the evaluations in this README file can be conducted on a machine with 4 cores and 64 GB of RAM.


**RoadMap:**
First, You should download the source code according to [Download Source Code](#download).
Then, you should set up the testing environment using docker according to [Set Up Environment](#environment).
Finally, after entering the docker contain, you can evalute the efficiency or accuracy of the frameworks according to [Efficiency Evaluation](#efficiency) and [Accuracy Evaluation](#accuracy), respectively.  We also provide solutions for the possible issues [Possible Issues](#issues).


## Download Source Code
<a name="download"></a>


You can download the source code by 

```
git clone https://github.com/FudanMPL/Garnet.git Ents -b Ents
```

or download the Ent.zip and unzip it.


## Set Up Environment
<a name="environment"></a>

We provide a dockerfile to set up the testing environment.

### 1. Change Working Directory

Use the following command to change current the working directory to Ents.

```
cd Ents/ 
```


### 2. Build Docker Image

Use the following command to build the Docker image:

```
sudo docker build -t ents .
```

Building the image may take some time.

Ignore the error info if the image are finally build successfully.

### 3. Launch Docker Container 

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
./Scripts/ring.sh -F -pn 10000 ents_efficiency-iris
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
./Scripts/ring.sh -F -pn 10000 hamada-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (3) Abspoel et al.'s Framework

To evaluate the efficiency of Abspoel et al.'s framework on the 'iris' dataset, execute these commands sequentially:


```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 abspoel iris
./Scripts/ring.sh -F -pn 10000 abspoel-iris
```


You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.

Note: Compiling for 'adult' and 'skin-segmentation' (i.e. runing the follow commands) may take 1-2 days and 400GB RAM.
```
python3 ./compile.py -R 128 abspoel adult (or skin-segmentation)
```


### 2. Ablation Experiment (Figure 2 in Section 5.3)

#### (1) Ents
To evaluate the efficiency of Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_efficiency iris
./Scripts/ring.sh -F -pn 10000 ents_efficiency-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (2) Hamada et al.'s Framework
To evaluate the efficiency of Hamada et al.'s framework on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada iris
./Scripts/ring.sh -F -pn 10000 hamada-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (3) Hamada et al.'s Framework-radixsort

To evaluate the efficiency of Hamada et al.'s framework-radixsort on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada_radixsort iris
./Scripts/ring.sh -F -pn 10000 hamada_radixsort-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


#### (4) Hamada et al.'s Framework-convert

To evaluate the efficiency of Hamada et al.'s framework-convert on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 hamada_convert iris
./Scripts/ring.sh -F -pn 10000 hamada_convert-iris
```

You can replace 'iris' with other datasets (cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation) to evaluate on other datasets.


### 3. Experiments for Share Conversion Protocol (Table 4 in Section 5.4)



#### (1) ConvertShare

To evaluate the efficiency of the protocol ConvertShare with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion 10
./Scripts/ring.sh -F -pn 10000 test-conversion-10
```
 
It will output as follows:

```
Using security parameter 40
Trying to run 128-bit computation
conversion with change domin size=10
The following benchmarks are excluding preprocessing (offline phase).
Time = 0.0162606 seconds 
Time1 = 0.00636278 seconds (0.000656 MB)
Data sent = 0.000688 MB in ~10 rounds (party 0)
Global data sent = 0.001424 MB (all parties)
```

The runtime and communication size are 0.00636278 seconds and 0.000656 MB, respectively (referred to as Time1). All subsequent evaluations in this section utilize Time1. It is important to note that the reported number of communication rounds (10 rounds) exceeds the actual required rounds (1 round) due to inaccuracies in the communication round counting method.

#### (2) Convert-A2B

To evaluate the efficiency of the protocol Convert-A2B with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion-with-a2b 10
./Scripts/ring.sh -F -pn 10000 test-conversion-with-a2b-10
```

#### (3) Convert-Dabits

To evaluate the efficiency of the protocol Convert-Dabits with vector of size n (such as 10), execute these commands sequentially:

```
python3 ./compile.py -R 128 test-conversion-with-dabits 10
./Scripts/ring.sh -F -pn 10000 test-conversion-with-dabits-10
```

#### (4) Convert-FSS

To evaluate the efficiency of the protocol Convert-FSS with vector of size n (such as 10), execute these commands sequentially:

```
./Fake-Offline.x 3 -e 15,31,63
make -j 6 fss-ring-party.x
python3 ./compile.py -R 128 test-conversion-with-fss 10
./Scripts/fss-ring.sh -F -pn 10000 test-conversion-with-fss-10
```



### 4. Experiment for Two-Party Ents (Table 6 in Appendix E.2) 
To  evaluate the efficiency of two-party Ents on the 'iris' dataset, execute these commands sequentially:

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_two_party iris
./Scripts/semi2k.sh -F -pn 10000 ents_two_party-iris
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


 
## Possible Issues
<a name="issues"></a>

### 1. Handshake Error

If you encounter the following error message, you may resolve it by running ```c_rehash ./Player-Data``` or by waiting some time before re-running the process.

```
Using security parameter 40
Trying to run 128-bit computation
Server-side handshake with P2 failed. Make sure both sides  have the necessary certificate (Player-Data/P0.pem in the default configuration on their side and Player-Data/P2.pem on ours), and run `c_rehash <directory>` on its location.
The certificates should be the same on every host. Also make sure that it's still valid. Certificates generated with `Scripts/setup-ssl.sh` expire after a month.
See also https://mp-spdz.readthedocs.io/en/latest/troubleshooting.html#handshake-failures
Signature (should match the other side): afa1084b09d095be77900966825d6fb7/cac0d81a218fbdbe0d3ed39d9ee1af85
terminate called after throwing an instance of 'boost::wrapexcept<boost::system::system_error>'
  what():  handshake: Connection reset by peer [system:104]
/usr/src/Ents/Scripts/run-common.sh: line 62:  1838 Aborted                 (core dumped) $prefix $SPDZROOT/$bin $i $params 2>&1
      1839 Done                    | { if test "$BENCH"; then
    if test $i = 0; then
        tee -a $log;
    else
        cat >> $log;
    fi;
else
    if test $i = 0; then
        tee $log;
    else
        cat > $log;
    fi;
fi; }
```

### 2. Binding Error

If you encounter the following error message, please terminate the process occupying port 10000 and try running the evluation again.  Additionally, note that all evaluations specified in the readme file utilize port 10000, and therefore cannot be executed simultaneously.


```
Default bit length: 127
Default security parameter: 40
(381, 1, 0, 0)
Compiling file /usr/src/Ents/Programs/Source/hamada_radixsort.mpc
Writing to /usr/src/Ents/Programs/Bytecode/hamada_radixsort-iris-multithread-1.bc
Writing to /usr/src/Ents/Programs/Bytecode/hamada_radixsort-iris-multithread-2.bc
Writing to /usr/src/Ents/Programs/Bytecode/hamada_radixsort-iris-multithread-3.bc
Writing to /usr/src/Ents/Programs/Bytecode/hamada_radixsort-iris-multithread-4.bc
Writing to /usr/src/Ents/Programs/Schedules/hamada_radixsort-iris.sch
Writing to /usr/src/Ents/Programs/Bytecode/hamada_radixsort-iris-0.bc
Program requires at most:
   410670450 online communication bits
   228312345 offline communication bits
       10367 online round
      321720 offline round
Running /usr/src/Ents/Scripts/../replicated-ring-party.x 0 -F -pn 10000 hamada_radixsort-iris -pn 16522 -h localhost
Running /usr/src/Ents/Scripts/../replicated-ring-party.x 1 -F -pn 10000 hamada_radixsort-iris -pn 16522 -h localhost
Running /usr/src/Ents/Scripts/../replicated-ring-party.x 2 -F -pn 10000 hamada_radixsort-iris -pn 16522 -h localhost
Using security parameter 40
Binding to socket on 4c9dfb475dce:10000 failed (Address already in use), trying again in a second ...
Binding to socket on 4c9dfb475dce:10000 failed (Address already in use), trying again in a second ...
Binding to socket on 4c9dfb475dce:10000 failed (Address already in use), trying again in a second ...
Binding to socket on 4c9dfb475dce:10000 failed (Address already in use), trying again in a second ...
...
terminate called after throwing an instance of 'std::runtime_error'
  what():  error in network setup: 4c9dfb475dce : Receiving error - 1 : Connection reset by peer
/usr/src/Ents/Scripts/run-common.sh: line 62:  1580 Aborted                 (core dumped) $prefix $SPDZROOT/$bin $i $params 2>&1
      1581 Done                    | { if test "$BENCH"; then
    if test $i = 0; then
        tee -a $log;
    else
        cat >> $log;
    fi;
else
    if test $i = 0; then
        tee $log;
    else
        cat > $log;
    fi;
fi; }
```