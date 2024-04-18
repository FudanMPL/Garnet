#!/usr/bin/env python3

import sys, operator, struct

try:
    filename = sys.argv[3]
except:
    filename = 'Player-Data/Private-Input-0'

out = open(filename, 'bw')

for line in open(sys.argv[1]):
    line = (line.strip())
    if line:
        x = (line.split(' '))
        for xx in x:
            out.write(struct.pack('<q', int(xx)))
