#!/usr/bin/env python3

import sys, os
import collections

sys.path.append('.')

from Compiler.program import *
from Compiler.instructions_base import *

if len(sys.argv) <= 1:
    print('Usage: %s <program>' % sys.argv[0])

res = collections.defaultdict(lambda: 0)
regs = collections.defaultdict(lambda: 0)

for tapename in Program.read_tapes(sys.argv[1]):
    for inst in Tape.read_instructions(tapename):
        t = inst.type
        if issubclass(t, DirectMemoryInstruction):
            res[t.arg_format[0]] = max(inst.args[1].i + inst.size,
                                       res[t.arg_format[0]])
        for arg in inst.args:
            if isinstance(arg, RegisterArgFormat):
                regs[type(arg)] = max(regs[type(arg)], arg.i + inst.size)

reverse_formats = dict((v, k) for k, v in ArgFormats.items())

print ('Memory:', dict(res))
print ('Registers:', dict((reverse_formats[t], n) for t, n in regs.items()))
