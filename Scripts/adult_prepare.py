#!/usr/bin/python3

import sys

binary = 'binary' in sys.argv
mixed = 'mixed' in sys.argv
nocap = 'nocap' in sys.argv

if binary:
    out = open('binary', 'w')
elif mixed:
    out = open('mixed', 'w')
elif nocap:
    out = open('nocap', 'w')
else:
    out = open('data', 'w')

for start, suffix in (0, 'data'), (1, 'test'):
    data = [l.strip().split(', ') for l in open('./Data/adult.%s' % suffix)][start:-1]

    print(' '.join(str(int(x[-1].startswith('>50K'))) for x in data), file=out)

    total = 0
    max_value = 0

    if not binary:
        if nocap:
            attrs = 0, 4, 12
        else:
            attrs = 0, 2, 4, 10, 11, 12
        for i in attrs:
            print(' '.join(x[i] for x in data), file=out)
            total += 1
            for x in data:
                max_value = max(int(x[i]), max_value)

    if binary or mixed or nocap:
        values = [set() for x in data[0][:-1]]
        for x in data:
            for i, value in enumerate(x[:-1]):
                values[i].add(value)
        for i in 1, 3, 5, 6, 7, 8, 9:
            x = sorted(values[i])
            print('Using attribute %d:' % i,
                  ' '.join('%d:%s' % (total + j, y)
                           for j, y in enumerate(x)))
            total += len(x)
            for y in x:
                print(' '.join(str(int(sample[i] == y)) for sample in data),
                      file=out)

    print(len(data), 'items')
    print(total, 'attributes')
    print('max value', max_value)
