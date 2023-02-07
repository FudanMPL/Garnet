#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

# number of players
players=${1:-2}
# prime field bit length
bits=${2:-128}
# binary field bit length, default by binary
g=${3:-0}
# default number of triples etc. to create
default=${4:-10000}

die () {
    echo >&2 "$@"
    echo >&2 "Usage:
setup-online.sh [nplayers] [prime_bitlength] [gf2n_bitlength] [num_prep]
Defaults:
nplayers=2, prime_bitlength=128, gf2n_bitlength=40/128 (as compiled), num_prep=10000"
    exit 1
}

[ "$#" -le 4 ] || die "More than 4 arguments provided"

for arg in "$@"
do
    echo "$arg" | grep -E -q '^[0-9]+$' || die "Integer argument required, $arg provided"
done

$HERE/setup-ssl.sh ${players}

$SPDZROOT/Fake-Offline.x ${players} -lgp ${bits} -lg2 ${g} --default ${default}

