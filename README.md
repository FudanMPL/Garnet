# ReadMe for Ents


The code for implementing Ents is primarily located in Compiler/ents.py and Protocols/replicated.hpp.




## Set Up Environment

We provide two ways to set up environment, one is using Docker, the other is setting up environment manually. We recommend to use Docker.


### 1. Use Docker

#### (1) build Docker images

```
sudo docker build -t ents .
```

This step may cost some time

#### (2) build Docker container

```
sudo docker run -it ents bash
```

#### (3) enter Docker container

```
docker build -t ents .
```


### 2. Build Environment Manually

If you choose to use Docker, ignore this part.

#### system requirement

ubuntu  20.04

python == 3.10.12

#### (1) library install

```
sudo apt-get install automake build-essential cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm
```

#### (2) python library install

```
pip install -r requirements.txt
```

#### (3) compile virtual machines


```
make clean
make -j 8 tldr
make -j 8 replicated-ring-party.x
make -j 8 semi2k-party.x
make -j 8 Fake-Offline.x
make -j 8 malicious-rep-ring-party.x
```

#### (4) setup certificate

```
./Scripts/setup-ssl.sh 3
```

#### (5) setup offline data

```
./Scripts/setup-online.sh 3 32
./Scripts/setup-online.sh 3 128
./Scripts/setup-online.sh 2 32
./Scripts/setup-online.sh 2 128
```




## Efficiency Evaluation

### 0. Network Simulation

All the efficiency evaluation are performed under two network setting: LAN and WAN.

To simulate the LAN setting: run the following codes
```
./lan.sh
```

To simulate the WAN setting: run the following codes
```
./wan.sh
```



### 1. Main Experiment (Table 3 in Section 5.2)

#### (1) Ents
To verify the efficiency of Ents on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_efficiency iris
./Scripts/ring.sh -F ents_efficiency-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation



#### (2) Hamada et al.'s framework
To verify the efficiency of Hamada et al.'s framework on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada iris
./Scripts/ring.sh -F hamada-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


#### (3) Abspoel et al.'s framework
To verify the efficiency of Abspoel et al.'s framework on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 abspoel iris
./Scripts/ring.sh -F abspoel-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation.
Note that compiling adult and skin-segmentation may need one or two days.

### 2. Ablation Experiment (Figure 2 in Section 5.3)

#### (1) Ents
To verify the efficiency of Ents on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 ents_efficiency iris
./Scripts/ring.sh -F ents_efficiency-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


#### (2) Hamada et al.'s framework
To verify the efficiency of Hamada et al.'s framework on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada iris
./Scripts/ring.sh -F hamada-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


#### (3) Hamada et al.'s framework-radixsort
To verify the efficiency of Hamada et al.'s framework-radixsort on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 128 hamada_radixsort iris
./Scripts/ring.sh -F hamada_radixsort-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


#### (4) Hamada et al.'s framework-extend
To verify the efficiency of Hamada et al.'s framework-extend on a dataset (such as iris), run the following codes

```
python3 ./Scripts/data_prepare_for_efficiency.py iris
python3 ./compile.py -R 32 hamada_extend iris
./Scripts/ring.sh -F hamada_extend-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


### 3. Share Conversion Protocol (Table 4 in Section 5.4)

#### (1) ConvertShare
To verify the efficiency of the protocol Convert with vector of size n (such as 10), run the following codes

```
python3 ./compile.py -R 128 test-conversion 10
./Scripts/ring.sh -F test-conversion-10
```



#### (2) Convert-A2B
To verify the efficiency of the protocol Convert-A2B with vector of size n (such as 10), run the following codes

```
python3 ./compile.py -R 128 test-conversion-with-a2b 10
./Scripts/ring.sh -F test-conversion-with-a2b-10
```

#### (3) Convert-Dabits
To verify the efficiency of the protocol Convert-Dabits with vector of size n (such as 10), run the following codes

```
python3 ./compile.py -R 128 test-conversion-with-dabits 10
./Scripts/ring.sh -F test-conversion-with-dabits-10
```

#### (4) Convert-FSS
To verify the efficiency of the protocol Convert with vector of size n (such as 10), run the following codes

```
./Fake-Offline.x 3 -e 15,31,63
make -j 6 fss-ring-party.x
python3 ./compile.py -R 128 test-conversion-with-fss 10
./Scripts/fss-ring.sh -F test-conversion-with-fss-10
```




#### 4. Table 6 in Appendix E.2
To verify the efficiency of two-party Ents on a dataset (such as iris), run the following codes

```
printf '0 %.0s' {1..10000000} >  Player-Data/Input-P0-0
python3 ./compile.py -R 32 ents_two_party iris
./Scripts/semi2k.sh -F ents_two_party-iris
```

## Accuracy evaluation (Table 7 in Appendix F)


#### (1) Ents

To verify the accuracy of Ents on iris, run the following codes:

```
python3 ./Scripts/data_prepare_for_accuracy iris
python3 ./compile.py -R 32 ents_accuracy iris
./Scripts/ring.sh -F ents_accuracy-iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation

Note that the accuracy may be a little different with the result in Table 7, that is because when multiple split points have the same maximum modified Gini impurity, Ents
will randomly select one.
 

#### (2) scikit-learn

To verify the accuracy of scikit-learn on iris, run the following codes:

```
python3 ./Scripts/ski-dt.py iris
```

iris coule be replaced by cancer, diagnosis, adult, kohkiloyeh, tic-tac-toe, wine or skin-segmentation


Note that the accuracy may be a little different with the result in Table 7, that is because when multiple split points have the same maximum modified Gini impurity, scikit-learn
will also randomly select one.
 