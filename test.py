import math
def sigmoid_clear(x):
    return 1/(1+math.exp(-x))

for i in range(-10,10):
    print(sigmoid_clear(i))

import random

