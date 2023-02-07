#!/usr/bin/env bash

while getopts t: opt; do
    case $opt in
	t) threshold=$OPTARG ;;
    esac
done

shift $[OPTIND-1]

gdb_screen()
{
    prog=$1
    shift
    screen -S :$1 -d -m bash -l -c "echo $*; echo $LIBRARY_PATH; gdb $prog -ex \"run $*\""
}

killall -9 bmr-party.x bmr-tparty.x memcheck-amd64-valgrind callgrind-amd64-valgrind bmr-program-party.x bmr-program-tparty.x gdb
dir=$(mktemp -d)
rm -R bmr-log
ln -s $dir bmr-log 2>/dev/null

ulimit -v $(grep MemTotal /proc/meminfo | awk '{printf "%d", $2/2}');
ulimit -c 0

prog=$1
n_players=${2:-2}

netmap=netmap.$(hostname)
echo $[n_players+1] > $netmap
for i in $(seq 0 $n_players); do
    echo 127.0.0.1 $[10000+i] >> $netmap
done

$prefix ./bmr-program-tparty.x $prog $netmap 2>&1 &> bmr-log/t &
for i in $(seq $[n_players-1]); do
    $prefix ./bmr-program-party.x $i $prog $netmap $threshold 2>&1 &> bmr-log/$i &
    id=$!
done
$prefix ./bmr-program-party.x $n_players $prog $netmap $threshold 2>&1 | tee bmr-log/$n_players
wait $id
