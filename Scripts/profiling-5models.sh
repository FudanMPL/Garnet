#!/usr/bin/env bash

echo 'densenet'
for i in {1..5}; do
    python compile.py -R 60 -Q CryptFlow2 -C -K LTZ,TruncPr autograd_densenet --profiling > densenet$i.txt
done

echo 'moblienet'
for i in {1..5}; do
    python compile.py -R 60 -Q CryptFlow2 -C -K LTZ,TruncPr autograd_mobilenetv3 --profiling > mobilenet$i.txt
done

echo 'resnet'
for i in {1..5}; do
    python compile.py -R 60 -Q CryptFlow2 -C -K LTZ,TruncPr autograd_resnet --profiling > resnet$i.txt
done

echo 'shufflenet'
for i in {1..5}; do
    python compile.py -R 60 -Q CryptFlow2 -C -K LTZ,TruncPr autograd_shufflenetv2 --profiling > shufflenet$i.txt
done

echo 'mpcformer'
for i in {1..5}; do
    python compile.py -R 64 -Q MPCFormer -C -K exp_fx,EQZ,FPDiv,Reciprocal autograd_MPCFormer --profiling > mpcformer$i.txt
done