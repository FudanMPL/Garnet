#!/usr/bin/env python3

import sys, os, csv
sys.path.insert(0, os.path.abspath('..'))

from Compiler import instructions_base as base
from Compiler import instructions
from Compiler.GC import instructions as gc
from itertools import chain
import re

desc = {}

for x in chain(instructions.__dict__.values(), gc.__dict__.values()):
    try:
        desc[x.code] = x.__doc__, x.__module__ + '.' + x.__name__
    except:
        pass

out = csv.writer(open('instructions.csv', 'w'))

items = list(chain(base.opcodes.items(), gc.opcodes.items()))

for name, opcode in sorted(items, key=lambda x: x[1]):
    d, n = desc.get(opcode, (None, None))
    if d and '$' not in d and '|' not in d and opcode not in \
       (0x65, 0x6a) :
        m = re.split(r'\.\s', d)
        if m:
            d = m[0]
        d = d.replace('\n', '')
        d = d.strip()
        out.writerow([':py:class:`%s <%s>`' % (name, n), hex(opcode), d])

del out
