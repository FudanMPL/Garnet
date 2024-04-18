#!/usr/bin/env bash

while getopts t: opt; do
    case $opt in
	t) threshold=$OPTARG ;;
    esac
done

shift $[OPTIND-1]

prog=$1
shift

get_ip()
{
    host $1 | grep address | cut -f 4 -d ' '
}

port_base=10000
netmap=netmap.$(hostname)

add_to_netmap()
{
    port=$[port_base+$2]
    if [[ $1 =~ ^[0-9].* ]]; then
	echo $1 $port >> $netmap
    else
	echo $(get_ip $1) $port  >> $netmap
    fi
}

i=0

echo $[$#+1] > $netmap
add_to_netmap $(hostname) 0

for host in $@; do
    i=$[i+1]
    add_to_netmap $host $i
done

for host in $@; do
    ssh $host killall -9 bmr-party.x bmr-tparty.x memcheck-amd64-valgrind callgrind-amd64-valgrind bmr-program-party.x bmr-program-tparty.x gdb
done

killall -9 bmr-party.x bmr-tparty.x memcheck-amd64-valgrind callgrind-amd64-valgrind bmr-program-party.x bmr-program-tparty.x

i=0
for host in $@; do
    i=$[i+1]
    #rsync $netmap "$host:$(pwd)"
    pwd=$pwd
    if test "$prefix" = gdb_screen; then
	screen -S :$i -d -m ssh $host "ulimit -c 0; ulimit -v $(grep MemTotal /proc/meminfo | awk '{print $2}'); cd $(pwd); pwd; gdb ./bmr-program-party.x -ex \"run $i $prog $netmap $threshold\"; sleep 100"
    else
	ssh $host "ulimit -c 0; ulimit -v $(grep MemTotal /proc/meminfo | awk '{print $2}'); cd $(pwd); pwd; { /usr/sbin/tc qdisc; $prefix ./bmr-program-party.x $i $prog $netmap $threshold; } 2>&1 | tee -a logs/bmr-$prog-t$threshold-N$#-$i-$host" & true
    fi
done

ulimit -c 0
ulimit -v $(grep MemTotal /proc/meminfo | awk '{printf "%d", $2/2}')

if test "$prefix" = gdb_screen; then
    screen -S :t -d -m bash -l -c "cd $(pwd); gdb ./bmr-program-tparty.x -ex \"run $prog $netmap\"; sleep 100"
else
    tlog=logs/bmr-$prog-t
    { echo netmap trusted; cat $netmap; ./bmr-program-tparty.x $prog $netmap; } 2>&1 | tee $tlog
fi
