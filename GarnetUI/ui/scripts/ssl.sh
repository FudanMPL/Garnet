#!/usr/bin/env bash

# brew-installed OpenSSL on MacOS
PATH=/usr/local/opt/openssl/bin:$PATH

name=${1:-"local"}
ssl_dir=${2:-"uploads/ssl"}

test -e $ssl_dir || mkdir $ssl_dir

openssl req -newkey rsa -nodes -x509 -out $ssl_dir/$name.pem -keyout $ssl_dir/$name.key -subj "/CN=$name" > /dev/null
