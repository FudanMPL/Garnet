import Compiler.tensor as tensor
from Compiler.tensor import *


def relu(input, inplace=False):
    pass


def relu_(input):
    pass


def gelu(input):
    pass


def sigmoid(input):
    pass


def logsigmoid(input):
    pass


def tanh(input):
    pass


def softmax(input, dim=None):
    pass


def log_softmax(input, dim=None):
    pass


def linear(input, weight, bias=None):
    pass


def conv2d(input, weight, bias=None, stride=1, padding=0):
    pass


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0):
    pass


def max_pool2d(input, kernel_size, stride=None, padding=0,):
    pass


def avg_pool2d(input, kernel_size, stride=None, padding=0,):
    pass


def dropout(input, p=0.5, training=True, inplace=False):
    pass


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    pass


def batch_norm(input, weight=None, bias=None, training=False, eps=1e-05):
    pass


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    pass


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    pass


def pdist(input, p=2):
    pass


def kl_div(input, target, log_target=False):
    pass


def l1_loss(input, target):
    pass


def nll_loss(input, target, weight=None):
    pass


def mse_loss(input, target):
    loss = (input - target).pow(2).mean()
    return loss


def binary_cross_entropy(input, target, weight=None):
    pass


def cross_entropy(input, target, weight=None):
    pass
