#!/bin/bash

source .venv/bin/activate
echo $1
cd $1
echo ${@:2}
${@:2}