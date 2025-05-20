#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

export PLAYERS=${PLAYERS:-3}

. $HERE/run-common.sh

run_player malicious-ccd-party.x $* || exit 1
