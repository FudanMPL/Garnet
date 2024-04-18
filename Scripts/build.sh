#!/usr/bin/env bash

function build
{
    echo ARCH = $1 >> CONFIG.mine
    echo GDEBUG = >> CONFIG.mine
    root=`pwd`
    cd deps/libOTe
    rm -R out
    python3 build.py --install=$root/local -- -DENABLE_SOFTSPOKEN_OT=ON -DBUILD_SHARED_LIBS=0 -DCMAKE_INSTALL_LIBDIR=lib $3
    cd $root
    make clean
    rm -R static
    mkdir static
    make -j 4 static-release || exit 1
    mkdir bin
    dest=bin/`uname`-$2
    rm -R $dest
    mv static $dest
    strip $dest/*
}

make deps/libOTe/libOTe

echo AVX_OT = 0 >> CONFIG.mine
build '-maes -mpclmul -DCHECK_AES -DCHECK_PCLMUL -DCHECK_AVX' amd64 -DENABLE_AVX=OFF

echo AVX_OT = 1 >> CONFIG.mine
build '-msse4.1 -maes -mpclmul -mavx -mavx2 -mbmi2 -madx -DCHECK_ADX' avx2 -DENABLE_AVX=ON
