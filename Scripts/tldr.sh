#!/usr/bin/env sh

if test `uname` = "Linux"; then
    flags='cat /proc/cpuinfo'
elif test `uname` = Darwin; then
    if ! type brew; then
	echo Do you want me to install Homebrew?
	echo Press RETURN to continue or any other key to abort
	read ans
	if test "$ans"; then
	    echo Aborting
	    exit 1
	else
	    /usr/bin/env ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	fi
    fi
    make tldr
else
    echo OS unknown
    exit 1
fi

if test "$flags"; then
    if $flags | grep -q avx2; then
	cpu=avx2
    else
	if test `uname -m` != x86_64; then
	    echo Binaries are not available for `uname -m`
	    echo Use the source distribution: https://github.com/data61/MP-SPDZ/#tldr-source-distribution
	    exit 1
	fi
	cpu=amd64
    fi

    if ! cp -av bin/`uname`-$cpu/* .; then
	echo This only works with a release downloaded from https://github.com/data61/MP-SPDZ/releases 1>&2
	echo Make sure NOT to download a source code only file 1>&2
	exit 1
    fi
fi

mkdir Player-Data 2> /dev/null
exit 0
