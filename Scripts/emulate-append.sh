#!/bin/bash

. $(dirname $0)/run-common.sh
prog=${1%.sch}
prog=${prog##*/}
shift
$prefix ./emulate.x $prog $* 2>&1 | tee -a logs/emulate-append-$prog
