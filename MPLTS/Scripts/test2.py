import numpy as np
import matplotlib.pyplot as plt
import time
import sys

data = np.loadtxt('ts.data')

labels = ['round_online', 'comm_bits_online', 'round_total', 'comm_bits_total', 'MPTS-cost']
cols = {}

for i in range(0,5):
    cols[labels[i]] = data[:,i]

for label in labels:
    print(label, min(cols[label]) / cols[label][0])