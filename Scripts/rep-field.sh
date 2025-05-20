#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

export PLAYERS=3

. $HERE/run-common.sh

run_player replicated-field-party.x $* || exit 1
