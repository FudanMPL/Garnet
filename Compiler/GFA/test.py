import math
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

res = []
def get_spline(precision):
    end_value = (1 << precision)
    for y in range(1, end_value):
        x = math.log(y/end_value)
        res.append(x)

get_spline(10)
X = []
Y = []
# print(res)
for i in res:
    X.append(i)
    Y.append(math.pow(math.e, i))
    


max_error = 0
for idx in range(len(X)):
    y = params[0] * X[idx] + params[1]
    y = y * X[idx] + params[2]
    max_error = max(max_error, abs(Y[idx]-y))
print(max_error)