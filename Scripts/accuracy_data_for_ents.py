#!/usr/bin/python3
import csv
import sys
import pandas as pd

data_name = sys.argv[1]


output_data = []
label = 0
attribute = 0
for suffix in 'train', 'test':
    file_name = './Data/%s_%s.csv' % (data_name, suffix)
    data = pd.read_csv(file_name, header=None)
    data = data.transpose()
    x = data[:-1]
    y = data[-1:]
    x = x.astype("int")
    y = y.astype("int")
    output_data = output_data + y.values.tolist()
    output_data = output_data + x.values.tolist()
    max_y = max(max(y.values.tolist()))
    label = max(max_y, label)
    attribute = len(data) - 1
    print("file:", file_name)
    print('items:', len(x.columns))

print('attributes:', len(data) - 1)
print('label:', label + 1)

output_file = open("./Player-Data/Input-P0-0", 'w')

for line in output_data:
    output_file.write(" ".join(str(i) for i in line))
    output_file.write("\n")
output_file.close()