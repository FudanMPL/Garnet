import Compiler.tensor as tensor
from Compiler.tensor import *


def relu(input, inplace=False):
    return input


def relu_(input):
    return input


def gelu(input):
    pass


def sigmoid(input):
    return input


def tanh(input):
    return input


def softmax(input):
    return input


def log_softmax(input):
    return input


def conv2d(input):
    return input


def max_pool(input):
    return input


def avg_pool(input):
    return input


def dropout(input):
    return input


def one_hot(input):
    return input


def normalize(input):
    return input


def batch_norm(input):
    return input


def layer_norm(input):
    return input


def kl_div(input):
    return input


def l1_loss(input):
    return input


def mse_loss(input, target, reduction='mean'):
    if reduction == 'mean':
        loss = (input - target).pow(2).mean()
    return loss


def binary_cross_entropy(input):
    return input


def cross_entropy(input):
    return input
