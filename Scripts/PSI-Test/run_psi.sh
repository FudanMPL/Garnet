#!/bin/bash

param=$1

case $param in
    "c")
        sudo ./compile.py -F 64 test_psi -v -C -K psi_cisc
        ;;
    "r")
        sudo ./compile.py -l -F 64 test_psi
        ;;
    "x")
        sudo make -j 8 semi2k-party.x
        ;;
    "0")
        sudo ./semi2k-party.x -N 2 -p 0 test_psi
        ;;
    "1")
        sudo ./semi2k-party.x -N 2 -p 1 test_psi
        ;;
esac