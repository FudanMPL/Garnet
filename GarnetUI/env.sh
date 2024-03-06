#!/bin/bash

# 从环境变量中读取四个变量
HostName=${HostName:-"GarnetUI"}
HostAddress=${HostAddress:-"localhost"}
HostPort=${HostPort:-8000}
GarnetPath=${GarnetPath:-"/Garnet"}

# 修改ui/ui/settings.py文件
sed -i "s|GARNETPATH = .*|GARNETPATH = \"$GarnetPath\"|" ui/ui/settings.py
sed -i "s|NAME = .*|NAME = \"$HostName\"|" ui/ui/settings.py
sed -i "s|IPADDRESS = .*|IPADDRESS = \"$HostAddress\"|" ui/ui/settings.py
sed -i "s|PORT = .*|PORT = $HostPort|" ui/ui/settings.py

# 修改Vue3-garnet-ui/src/utils/request.js文件
sed -i "6s|const baseURL = .*|const baseURL = 'http://$HostAddress:$HostPort/'|" Vue3-garnet-ui/src/utils/request.js

