import Compiler.tensor as tensor
from Compiler.tensor import *


def relu(input, inplace=False): #todo
    pass

def gelu(input): #todo low priority
    pass


def sigmoid(input): #todo
    pass


def logsigmoid(input): #todo
    pass


def tanh(input): #todo
    pass


def softmax(input, dim=None): #todo
    pass


def log_softmax(input, dim=None): #todo
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


def dropout(input, p=0.5, training=True, inplace=False): #todo
    pass


def one_hot(input, num_classes=-1):
    # i think user should specify the num_classes, if not, we should calculate the max value in input.
    """example:
    one_hot(torch.tensor([0, 1, 2, 3, 4]), num_classes=8)
    tensor([[1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0]])"""
    in_sizes = input.sizes
    output = MultiArray([*in_sizes, num_classes], input.values.value_type) # TODO: check if value_type is cint or int
    pass


def embedding(input, weight):
    pass


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None): #todo
    pass


def batch_norm(input, weight=None, bias=None, training=False, eps=1e-05):
    pass


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    pass


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    pass


def pdist(input, p=2): #todo
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
