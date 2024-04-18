#!/bin/bash

Scripts/adult_prepare.py mixed

out_dir=Player-Data
test -e $out_dir || mkdir $out_dir

cp mixed $out_dir/Input-P0-0
