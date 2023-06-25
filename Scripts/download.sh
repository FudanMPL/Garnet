#!/bin/bash

wget -nc -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{data,test} || exit 1
