#!/usr/bin/python3
import csv
import sys
import pandas as pd
import os

data_name = sys.argv[1]

current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.abspath(os.path.join(current_dir, "..", "Data", "data.txt"))

output_data = []
for suffix in 'train', 'test':
    # file_name = './UnitTest/Data/%s_%s.csv' % (data_name, suffix)
    file_name  = os.path.abspath(os.path.join(current_dir, "..", "Data", "%s_%s.csv" % (data_name, suffix)))
    data = pd.read_csv(file_name, header=None)
    data = data.transpose()
    x = data[:-1]
    y = data[-1:]
    x = x.astype("int")
    y = y.astype("int")
    output_data = output_data + y.values.tolist()
    output_data = output_data + x.values.tolist()


    print("file:", file_name)
    print('items:', len(x.columns))
    print('attributes:', len(data) - 1)



# output_file = open("./Player-Data/Input-P0-0", 'w')
output_file = open(os.path.abspath(os.path.join(current_dir, "..", "..", "Player-Data", "Input-P0-0")), 'w')


for line in output_data:
    output_file.write(" ".join(str(i) for i in line))
    output_file.write("\n")
output_file.close()