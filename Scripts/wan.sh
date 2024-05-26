sudo tc qdisc del dev lo root 
sudo tc qdisc add root dev lo handle 1: htb default 7
sudo tc class add dev lo parent 1: classid 1:1 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:2 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:3 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:4 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:5 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:6 htb rate 5Mbps
sudo tc class add dev lo parent 1: classid 1:7 htb rate 1000Mbps
sudo tc qdisc add dev lo parent 1:1 handle 2:0 netem delay 200ms
sudo tc qdisc add dev lo parent 1:2 handle 3:0 netem delay 200ms
sudo tc qdisc add dev lo parent 1:3 handle 4:0 netem delay 200ms
sudo tc qdisc add dev lo parent 1:4 handle 5:0 netem delay 200ms
sudo tc qdisc add dev lo parent 1:5 handle 6:0 netem delay 200ms
sudo tc qdisc add dev lo parent 1:6 handle 7:0 netem delay 200ms
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10000 0xffff flowid 1:1
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10000 0xffff flowid 1:2
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10001 0xffff flowid 1:3
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10001 0xffff flowid 1:4
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10002 0xffff flowid 1:5
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10002 0xffff flowid 1:6
