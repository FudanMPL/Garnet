#!/bin/bash

source .venv/bin/activate
cd $1
echo ${@:2}
${@:2}