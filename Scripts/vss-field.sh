#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

export PLAYERS=3

. $HERE/run-common.sh

run_player vss-field-party.x -ND 1 -NA 2 -NP 1 $* || exit 1