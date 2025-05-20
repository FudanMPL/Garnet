#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

. $HERE/run-common.sh

run_player semi-bmr-party.x $* || exit 1
