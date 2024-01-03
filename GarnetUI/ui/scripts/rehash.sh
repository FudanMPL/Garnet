#!/usr/bin/env bash

ssl_dir=$1
garnet=$2


rm $garnet/Player-Data/*.pem
rm $garnet/Player-Data/*.key

part=$3
local_name=$4
ln $ssl_dir/$local_name.pem $garnet/Player-Data/P$part.pem
ln $ssl_dir/$local_name.key $garnet/Player-Data/P$part.key
paramCount=$#

for (( i = 5; i <= paramCount; i+=2 )); do
    p=${!i}
    ii=$(($i+1))
    n=${!ii}
    ln $ssl_dir/$n.pem $garnet/Player-Data/P$part.pem
done
c_rehash $garnet/Player-Data