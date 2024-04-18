#!/usr/bin/env python3

import sys, os
import collections

sys.path.append('.')

from Compiler.program import *
from Compiler.instructions_base import *

if len(sys.argv) <= 1:
    print('Usage: %s <program>' % sys.argv[0])

res = collections.defaultdict(lambda: 0)
m = 0

tapename = next(Program.read_tapes(sys.argv[1]))
res = Tape.ReqNum()
for inst in Tape.read_instructions(tapename):
    res.update(inst.get_usage())

for x in res.pretty():
    print(x)
