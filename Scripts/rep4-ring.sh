#!/bin/bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

export PLAYERS=4

. $HERE/run-common.sh

run_player rep4-ring-party.x $* || exit 1
