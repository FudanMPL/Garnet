#!/usr/bin/env bash

ssl_dir=$1
garnet=$2


rm $garnet/Player-Data/*.pem
rm $garnet/Player-Data/*.key
rm $garnet/Player-Data/server.name
part=$3
local_name=$4
cp $ssl_dir/$local_name.key $garnet/Player-Data/P$part.key
paramCount=$#

for (( i = 5; i <= paramCount; i+=2 )); do
    p=${!i}
    ii=$(($i+1))
    n=${!ii}
    cp $ssl_dir/$n.pem $garnet/Player-Data/P$p.pem
    echo $n >> $garnet/Player-Data/server.name
done
c_rehash $garnet/Player-Data