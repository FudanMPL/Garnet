#!/usr/bin/env python3

import sys, operator

try:
    f = int(sys.argv[2])
except:
    f = 12

try:
    filename = sys.argv[3]
except:
    filename = 'Player-Data/Input-P0-0'

out = open(filename, 'w')

for line in open(sys.argv[1]):
    line = (line.strip())
    if line:
        x = (line.split(' '))
        out.write(' '.join(str(int(xx) / 2**f) for xx in x))
    out.write('\n')
