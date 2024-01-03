#!/usr/bin/env bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=$HERE/..

export PLAYERS=3

. $HERE/run-common.sh

run_player rss-with-conversion-party.x $* || exit 1
