from gfapp import GFA
import sympy as sp
import numpy as np
import math

# constant factors
PAI = 3.1415926
TAU_2 = 0.959502
ALPHA1 = 1.0
ALPHA2 = 1.6732632
LAMBDA = 1.0507010
E = 2.7182818
C1 = 0.044715
TAU_half = 1.7725
G = 0.5

# fundenmental functions.
def func_reciprocal(x):
    return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

def func_sqrt(x, lib=sp):
    return lib.sqrt(x)

def func_log(x, lib=sp):
    return lib.log(x)

def func_pow(a, x):
    return a**x

# sigmoid-approx
@GFA(10, 44, 96, (-10,10))
def sigmoid(x):
    return 1 * func_reciprocal((1 + func_exp(-x)))

# tanh-approx
@GFA(10, 44, 96, (-50,50))
def tanh(x):
    ep = func_exp(x)
    en = func_exp(-x)
    return (ep - en) * func_reciprocal(ep + en)

# softplus-approx
@GFA(15, 44, 96, (-20,50))
def soft_plus(x):
    return func_log(1 + func_exp(x))

# elu-approx
@GFA(10, 44, 96, (-50,20), False)
def elu(x):
    """Reference: https://arxiv.org/pdf/1511.07289.pdf
    """
    pos_flag = x > 0
    res = x * pos_flag + (1 - pos_flag) * ALPHA1 * (func_exp(x, lib=np) -1)
    return res

# selu-approx
@GFA(10, 44, 96, (-50,20), False)
def selu(x):
    """Reference: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
    """
    pos_flag = x > 0
    res = LAMBDA * x * pos_flag + (1 - pos_flag) * LAMBDA * (
        ALPHA2 * func_exp(x, lib=np) - ALPHA2)
    return res

# gelu-approx
@GFA(10, 44, 96, (-20,20), False)
def gelu(x):
    constant = math.sqrt(2 / PAI)
    x1 = constant * (x + C1 * x * x * x)
    ep = func_exp(x1, lib=np)
    en = func_exp(-x1, lib=np)
    return 0.5 * x * (1 + ((ep - en) * func_reciprocal(ep + en)))
    
# normal-distribution
@GFA(10, 44, 96, (-10,10))
def snormal_dis(x):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
    """
    return func_exp(((-x**2) / 2)) * func_reciprocal(func_sqrt(2 * PAI))

