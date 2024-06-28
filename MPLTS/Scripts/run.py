import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def draw():
    data = np.loadtxt('ts.data')

    labels = ['round_online', 'comm_bits_online', 'round_total', 'comm_bits_total', 'MPTS-cost']
    cols = {}

    for i in range(0,5):
        cols[labels[i]] = data[:,i]

    def max_zero_normalize(column):
        return column / column[0]

    for label in labels:
        cols[label] = max_zero_normalize(cols[label]) * 100

    plt.figure(figsize=(10, 6))

    for label in labels:
        plt.plot(cols[label], label=label)

    plt.title('MPL DNN Model Graph Optimizing')
    plt.xlabel('searching graphs')
    plt.ylabel('opt % of origin graph')
    plt.legend()

    plt.savefig('MPLTS.png')
    plt.close()

symbol = '/'
while 1:
    draw()
    
    symbol = '/' if symbol == '\\' else '\\'
    sys.stdout.write('\r>> MPTS is running ' + symbol)
    sys.stdout.flush()
    
    # 等待5秒
    time.sleep(2)
    
    