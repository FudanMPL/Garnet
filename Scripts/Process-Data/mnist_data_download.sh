#!/bin/bash
cd ./Data
wget -nc http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz || exit 1

for i in *.gz; do
    gunzip $i
done

wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz || exit 1

tar xzvf cifar-10-python.tar.gz