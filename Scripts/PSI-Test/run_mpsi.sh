#!/bin/bash

param=$1

case $param in
    "c")
        ./compile.py -F 64 test_mpsi
        ;;
    "x")
        sudo make -j 8 replicated-ring-party.x
        ;;
    "0")
        sudo ./replicated-ring-party.x -I 0 test_mpsi
        ;;
    "1")
        sudo ./replicated-ring-party.x -I 1 test_mpsi
        ;;
    "2")
        sudo ./replicated-ring-party.x -I 2 test_mpsi
        ;;
    "t")
        sudo bash Scripts/semi2k.sh test_psi > Scripts/PSI-Test/out.txt
esac