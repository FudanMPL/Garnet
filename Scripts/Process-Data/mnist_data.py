#!/usr/bin/python3

import struct, sys

w = lambda x: struct.unpack('>i', x.read(4))[0]
b = lambda x: struct.unpack('B', x.read(1))[0]

try:
    max_n = int(sys.argv[1])
except:
    max_n = None

try:
    scale = int(sys.argv[2])
except:
    scale = True

out = open("Player-Data/Input-P0-0",'w')

n_train = 1000
n_test = 100

for s in 'train', 't10k':
    labels = open('./Data/%s-labels-idx1-ubyte' % s, 'rb')
    images = open('./Data/%s-images-idx3-ubyte' % s, 'rb')

    assert w(labels) == 2049
    n_labels = w(labels)

    assert w(images) == 2051
    n_images = w(images)
    assert n_labels == n_images
    assert w(images) == 28
    assert w(images) == 28

    print ('%d total examples' % n_images, file=sys.stderr)

    data = []
    n = [0] * 10

    for i in range(n_images if max_n is None else min(max_n, n_images)):
        label = b(labels)
        image = [b(images) / 256 if scale else b(images)
                 for j in range(28 ** 2)]
        data.append(image)
        n[label] += 1
        l = [0] * 10
        l[label] = 1
        print(' '.join(str(x) for x in l), end=' ',file=out)
    print(file=out)

    print ('%d used examples %s' % (len(data), n), file=sys.stderr)

    for x in data:
        for y in x:
            print(y, end=' ',file=out)
        print(file=out)