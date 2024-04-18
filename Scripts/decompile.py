#!/usr/bin/env python3

import sys, os

sys.path.append('.')

from Compiler.instructions_base import Instruction
from Compiler.program import *

if len(sys.argv) <= 1:
    print('Usage: %s <program>' % sys.argv[0])

for tapename in Program.read_tapes(sys.argv[1]):
    with open('Programs/Bytecode/%s.asm' % tapename, 'w') as out:
        for i, inst in enumerate(Tape.read_instructions(tapename)):
            print(inst, '#', i, file=out)
