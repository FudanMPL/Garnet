from tensor import get_opid, Tensor, get_prepare, Operation, tensors, gradient_operation, op_id_store,fake_propagate, set_opid,dl_d
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
approx = False
def relu(input, inplace=False):  # todo
    op_id = get_opid()
    global tensors
    global gradient_operation
    global op_id_store
    global dl_d
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output_ = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=(input_.value[:]>=0)*dl_dy[:]
            
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        output.value[:] = (0 < input.value[:]).if_else(input.value[:], 0) 
        set_opid(op_id+1)  # record the input and output of the op
    return output

@vectorize
def approx_sigmoid(x, n=3):
    """ Piece-wise approximate sigmoid as in
    `Hong et al. <https://arxiv.org/abs/2002.04344>`_

    :param x: input
    :param n: number of pieces, 3 (default) or 5
    """
    if n == 5:
        cuts = [-5, -2.5, 2.5, 5]
        le = [0] + [x <= cut for cut in cuts] + [1]
        select = [le[i + 1] - le[i] for i in range(5)]
        outputs = [cfix(10 ** -4),
                   0.02776 * x + 0.145,
                     * x + 0.5,
                   0.02776 * x + 0.85498,
                   cfix(1 - 10 ** -4)]
        return sum(a * b for a, b in zip(select, outputs))
    else:
        a = x < -0.5
        b = x > 0.5
        return a.if_else(0, b.if_else(1, 0.5 + x))

def gelu(input):  # todo low priority
    pass

def log_e(x):
    return mpc_math.log_fx(x, math.e)

use_mux = False
def exp(x):
    if use_mux:
        return mpc_math.mux_exp(math.e, x)
    else:
        return mpc_math.pow_fx(math.e, x)

def get_limit(x):
    exp_limit = 2 ** (x.k - x.f - 1)
    return math.log(exp_limit)

def sanitize(x, raw, lower, upper):
    limit = get_limit(x)
    res = (x > limit).if_else(upper, raw)
    return (x < -limit).if_else(lower, res)

def sigmoid_from_e_x(x,e_x):
    return sanitize(x, 1 / (1 + e_x), 0, 1)

def sigmoid(input): #todo
    op_id = get_opid()
    global tensors
    global gradient_operation
    global op_id_store
    global dl_d
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output_ = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=output_.value[:]*(1-output_.value[:])*dl_dy[:]
            
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if approx:
            output.value[:]=approx_sigmoid(input.value[:])
        else:
            output.value[:] =  sigmoid_from_e_x(input.value[:],exp(-input.value[:]))
        set_opid(op_id+1)  # record the input and output of the op
    return output


def logsigmoid(input):  # todo
    op_id = get_opid()
    global tensors
    global gradient_operation
    global op_id_store
    global dl_d
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output_ = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=1/(output_.value[:]*(1-output_.value[:]))*dl_dy[:]
            
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        output.value[:] =  log_e(sigmoid_from_e_x(input.value[:],exp(-input.value[:])))
        set_opid(op_id+1)  # record the input and output of the op
    return output


def tanh(input):  # todo
    op_id = get_opid()
    global tensors
    global gradient_operation
    global op_id_store
    global dl_d
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output_ = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=(1-output.value[:]*output.value[:])*dl_dy[:]
            
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        x=input.value[:]
        ex=exp(x)
        e_x=exp(-x)
        output.value[:] = sanitize(x, (ex-e_x)/(ex+e_x), -1, 1)    
        set_opid(op_id+1)  # record the input and output of the op
    return output
    


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
    assert isinstance(input, Tensor), "input should be Tensor"
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
