import Compiler.tensor as tensor
from glob import glob
import math
import re
import numpy as np
# from turtle import forward, shape
from itertools import zip_longest
from Compiler import mpc_math, util
from Compiler.types import *
from Compiler.types import _unreduced_squant
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
# from Compiler.GC.types import sbitintis_train
from functools import reduce
from typing import List, NamedTuple, Callable, Dict, Optional, Union, Tuple, Any


def relu(input, inplace=False):  # todo
    pass


def gelu(input):  # todo low priority
    pass


def sigmoid(input): #todo
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input = tensor.tensors[operation.inputs[0]]
        output = tensor.tensors[operation.outputs[0]]
        if input.req_grad:
            dl_dy[:]+=(1/(1+mpc_math.exp_fx(input.value[:])))
            # dl_dy.mul_trans_add_to(input2.value,dl_d[operation.inputs[0]],n_threads=10 if input1.shape[0]>=1000 else 1)
            # C=AB partial derivate of dA=dC*B^T+dA
    # tensor.train()
    
    prepare = tensor.get_prepare()
    print(prepare)
    print(type(input))
    
    if prepare:
        print(type(input.value))
        assert isinstance(input.value, MultiArray),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
        output = tensor.Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = tensor.Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = tensor.Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        tensor.gradient_operation.append(operation)
        operation_id = len(tensor.gradient_operation) - 1
        tensor.op_id_store[tensor.op_id] = operation_id
        tensor.op_id += 1
    else:
        print(22222222)
        operation = tensor.gradient_operation[tensor.op_id_store[tensor.op_id]]
        input = tensor.tensors[operation.inputs[0]]
        output = tensor.tensors[operation.outputs[0]]
        output.value[:]=1/(1+mpc_math.exp_fx(input.value[:]))
        tensor.op_id += 1  # record the input and output of the op
    return output


def logsigmoid(input):  # todo
    pass


def tanh(input):  # todo
    pass


def softmax(input, dim=None):  # todo
    pass


def log_softmax(input, dim=None):  # todo
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


def dropout(input, p=0.5, training=True, inplace=False):  # todo
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
    assert input.value.value_type == cint, "input should be cint"
    x = input.value
    in_sizes = x.sizes
    b = reduce(operator.mul, in_sizes) if len(in_sizes) >= 2 else in_sizes[0]
    output = MultiArray([*in_sizes, num_classes], x.value_type)

    output.view(-1, num_classes)

    for i in range(b):
        output[i][x.get_vector()[i]] = 1

    output.view(*in_sizes, num_classes)
    return Tensor(output)


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):  # todo
    pass


def batch_norm(input, weight=None, bias=None, training=False, eps=1e-05):
    pass


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    pass


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    pass


def pdist(input, p=2):  # todo
    pass


def kl_div(input, target, log_target=False):
    pass


def l1_loss(input, target):
    pass


def nll_loss(input, target, weight=None):
    pass


def mse_loss(input, target): # todo
    pass


def binary_cross_entropy(input, target, weight=None):
    pass


def cross_entropy(input, target, weight=None):
    pass
