# MPTS

MPTS is the extention of TASO for graph optimizing in multi-party learning

* Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. [TASO: Optimizing Deep Learning Computation with Automated Generation of Graph Substitutions](https://cs.stanford.edu/~zhihao/papers/sosp19.pdf). In Proceedings of the Symposium on Operating Systems Principles (SOSP), Ontario, Canada, October 2019.

* Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, and Alex Aiken. [Optimizing DNN Computation with Relaxed Graph Substitutions](https://theory.stanford.edu/~aiken/publications/papers/sysml19b.pdf). In Proceedings of the Conference on Systems and Machine Learning (SysML), Palo Alto, CA, April 2019.

## Build

- CMAKE 3.2 or higher
- ProtocolBuffer 3.6.1 or higher
- Cython 0.28 or higher
- ONNX 1.5 or higher
- CUDA 11.7 or higher and CUDNN 7.0 or higher

and then add the following line in ~/.bashrc.
```
export TASO_HOME=/path/to/MPLTS
export Garnet_HOME=/path/to/Garnet
export LD_LIBRARY_PATH="/path/to/lib/:$LD_LIBRARY_PATH"
```

```
./MPLTS/recompile.sh
```


## Run example
```
python MPLTS/examples/resnet50.py  -R 64 -Q ABY3
python MPLTS/examples/resnext50.py -R 64 -Q ABY3
python MPLTS/examples/inceptionv3.py -R 64 -Q ABY3
python MPLTS/examples/nasnet_a.py -R 64 -Q ABY3
python MPLTS/examples/batched_resnet.py
```


## Script: Show figure
```
python MPLTS/experiments/run.py
```

## Script: Check
set up the [running environment](#appendix), check the result by running on the virtual machine.
```
python MPLTS/experiments/check.py -R 64
```


## Appendix <a name="appendix"></a>
two-party environment
```
make -j 8 tldr
make -j 8 semi2k.x
make -j 8 Fake-Offline.x
./Scripts/setup-ssl.sh 2
```
three-party environment
```
make -j 8 tldr
make -j 8 replicated-ring-party.x
make -j 8 Fake-Offline.x
./Scripts/setup-ssl.sh 3
```
