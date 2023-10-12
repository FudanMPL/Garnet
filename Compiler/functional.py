# from tensor import get_opid, Tensor, get_prepare, Operation, tensors, gradient_operation, op_id_store,fake_propagate, set_opid,dl_d
from tensor import *
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
approx = True
def relu(input, inplace=False):  # todo
    op_id = get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
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
def approx_sigmoid(x, n=5):
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
                   0.17 *x + 0.5,
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
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        # if input_.req_grad:
        dl_d[input_.name]+=output.value[:]*(1-output.value[:])*dl_dy[:]
            
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
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=1/(1+exp(output.value[:]))*dl_dy[:]
            
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
        output.value[:] = -log_e(1+exp(-input.value[:]))
        set_opid(op_id+1)  # record the input and output of the op
    return output


def tanh(input):  # todo
    op_id = get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
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
    assert isinstance(input,Tensor),"Invalid input"
    assert isinstance(weight,Tensor),"Invalid weight"
    assert input.shape[-1]==weight.shape[0],"Invalid Dimension"
    if len(input.sizes) > len(weight.sizes):
        output=input.single_bmm(weight)
    elif len(input.sizes) == len(weight.sizes):
        output=input.mm(weight)
    else:
        raise CompilerError("the dimension of input must not smaller than the dimension of weight")
    if bias is None:
        pass
    else:
        output = bias + output
    return output


def new_squant():
        class _(sfix):
            params = None
        return _


def conv2d(input:Tensor, weight:Tensor, bias=None, stride=[1,1], padding=[0,0]):
    #input.shape:(batch_size,channel_in,H,W)
    #weight.shape:(out_channels, in_channels // groups, H,W)
    #bais:(out_channels)
    op_id = get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input = tensors[operation.inputs[0]]
        weight= tensors[operation.inputs[1]]
        output = tensors[operation.outputs[0]]
        _, _,weights_h, weights_w= weight.shape
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        input_value=input.value.permute([0,2,3,1])
        weight_value=weight.value.permute([0,2,3,1])
        nabla_Y=output.grad.permute([0,2,3,1])

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        
        n_threads=8 if input.numel() > 2**20 else 1
        batch=Array.create_from(regint.inc(N))
        input_size = inputs_h * inputs_w * N #why have no channel_in? 128*36
        batch_repeat = regint.Matrix(N, inputs_h * inputs_w) # 128,6*6
        batch_repeat.assign_vector(batch.get(
            regint.inc(input_size, 0, 1, 1, N)) *
                                   reduce(operator.mul, input_value.sizes[1:]))
        @for_range_opt_multithread(n_threads, [n_channels_in, n_channels_out])
        def _(i, j):
            a = regint.inc(input_size, input_value.address + i, n_channels_in, N,
                           inputs_h * inputs_w)
            inputs = sfix.load_mem(batch_repeat.get_vector() + a).v
            b = regint.inc(N * output_w * output_h, nabla_Y.address + j, n_channels_out, N)
            rep_out = regint.inc(output_h * output_w * N, 0, 1, 1, N) * \
                reduce(operator.mul, nabla_Y.sizes[1:])
            nabla_outputs = sfix.load_mem(rep_out + b).v
            res = sint(size = weights_h * weights_w)
            conv2ds(res, inputs, nabla_outputs, weights_h, weights_w, inputs_h,
                    inputs_w, output_h, output_w, -stride_h, -stride_w, N,
                    padding_h, padding_w, 1) 
            reduced = unreduced_sfix._new(res).reduce_after_mul()
            weight.grad.assign_vector_by_indices(reduced, j, i,None, None)
        
        
        nabla_X=input.grad.permute([0,2,3,1])
        
        
        reverse_weights = MultiArray(
                [n_channels_in, weights_h, weights_w, n_channels_out], sfix)
        @for_range_opt_multithread(n_threads, n_channels_in)
        def _(l):
            @for_range(weights_h)
            def _(j):
                @for_range(weights_w)
                def _(k):
                    addresses = regint.inc(n_channels_out,
                        weight_value[0][j][weights_w-k-1].get_address(l),
                        reduce(operator.mul, weight_value.sizes[1:]))
                    reverse_weights[l][weights_h-j-1][k].assign_vector(
                        weight_value.value_type.load_mem(addresses))
        padded_w = inputs_w + 2 * padding_w
        padded_h = inputs_h + 2 * padding_h
        if padding_h or padding_w:
            output = MultiArray(
                [N, padded_h, padded_w, n_channels_in], sfix)
        else:
            output = nabla_X
        @for_range_opt_multithread(n_threads,
                                    [N, n_channels_in])
        def _(i, j):
            res = sint(size = (padded_w * padded_h))
            conv2ds(res, nabla_Y[i].get_vector().v,
                    reverse_weights[j].get_vector().v,
                    padded_h, padded_w, output_h, output_w,
                    weights_h, weights_w, 1, 1, n_channels_out,
                    weights_h - 1, weights_w - 1, 1)
            input.grad.assign_vector_by_indices(
                unreduced_sfix._new(res).reduce_after_mul(),i, j,None, None)
        if padding_h or padding_w:
            @for_range_opt_multithread(n_threads, N)
            def _(i):
                @for_range(inputs_h)
                def _(j):
                    @for_range(inputs_w)
                    def _(k):
                        jj = j + padding_w
                        kk = k + padding_w
                        nabla_X[i][j][k].assign_vector(output[i][jj][kk].get_vector())
            # nable_X.print_reveal_nested()
        
        
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor) and isinstance(weight, Tensor) ,"Invalid Input and weight"
        assert len(input.shape)==4 and len(weight.shape)==4,"Invalid Dimension input and weight"
        out_shape=[input.shape[0],weight.shape[0],(input.shape[2]+2*padding[0]-weight.shape[2])//stride[0]+1,
                   (input.shape[3]+2*padding[1]-weight.shape[3])//stride[1]+1] #out_shape.size:[Batch_size,out_channel,H_out,W_out]
        new_value=MultiArray(out_shape,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name,weight.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name,weight.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        operation = gradient_operation[op_id_store[op_id]]
        output= tensors[operation.outputs[0]] 
        _, _,weights_h, weights_w= weight.shape
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        input_value=input.value.permute([0,2,3,1])
        weight_value=weight.value.permute([0,2,3,1])
        output_value=output.value.permute([0,2,3,1])
        
        n_threads=8 if input.numel() > 2**20 else 1
        
        n_parts = max(1, round((n_threads or 1) / n_channels_out))
        while N % n_parts != 0:
            n_parts -= 1
        print('Convolution in %d parts' % n_parts)
        unreduced = MultiArray(output_value.sizes, sint, address=output_value.address)
        part_size =N // n_parts
        size_=part_size*reduce(operator.mul,input.shape[1:])
        @for_range_multithread(n_threads, 1, [n_parts, n_channels_out])
        def _(i, j):
            inputs = input_value.get_vector(i*size_,size_).v
            weights = weight_value.get_part_vector(j).v
            res = sint(size = output_h * output_w * part_size)
            conv2ds(res, inputs, weights, output_h, output_w,
                    inputs_h, inputs_w, weights_h, weights_w,
                    stride_h, stride_w, n_channels_in, padding_h, padding_w,
                    part_size)
            if bias:
                res += bias.expand_to_vector(j, res.size).v
            addresses = regint.inc(res.size,
                                    unreduced[i * part_size].address + j,
                                    n_channels_out)
            res.store_in_mem(addresses)
        n_outputs = N * reduce(operator.mul, output_value.sizes[1:])
        @multithread(n_threads, n_outputs,
                     1000 if sfix.round_nearest else 10 ** 6)                                                                                
        def _(base, n_per_thread):
            res = sfix().unreduced(sint.load_mem(unreduced.address + base,
                              size=n_per_thread),sfix).reduce_after_mul()
            res.store_in_mem(output.value.address + base)
        
        set_opid(op_id+1)  # record the input and output of the op
    return output


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, outputpadding=0):
     pass


def max_pool2d(input, kernel_size=2, stride=2, padding=0):
    op_id=get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        strides=[1]+list(operation.intermediate[0])+[1]
        ksize=[1]+list(operation.intermediate[1])+[1]
        n_threads=8 if input.numel() > 2**20 else 1
        
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        
        batch=Array.create_from(regint.inc(N))
        def process(pool, bi, k, i, j,comparisons,nabla_Y,nabla_X):
            for (x, h_in, w_in, h, w), c \
                in zip(pool, comparisons[bi][k][i][j]):
                hh = h * h_in
                ww = w * w_in
                res = h_in * w_in * c * nabla_Y[bi][k][i][j]
                nabla_X[bi][k][hh][ww] += res
        
        Y_sizes =[N,output_h, output_w,n_channels_out]  
        X_sizes =[N,inputs_h, inputs_w,n_channels_in]
        need_padding = [strides[i] * (Y_sizes[i] - 1) + ksize[i] >
                        X_sizes[i] for i in range(4)]
        overlap = reduce(operator.or_,
                         (x < y for x, y in zip(strides, ksize)))
        @for_range_opt_multithread(n_threads,
                                   [len(batch), n_channels_in])
        def _(l, k):
            bi = batch[l]
            @for_range_opt(output_h)
            def _(i):
                h_base = strides[1] * i
                @for_range_opt(output_w)
                def _(j):
                    if overlap:
                        break_point()
                    w_base = strides[2] * j
                    pool = []
                    for ii in range(ksize[1]):
                        h = h_base + ii
                        if need_padding[1]:
                            h_in = h < X_sizes[1]
                        else:
                            h_in = True
                        for jj in range( ksize[2]):
                            w = w_base + jj
                            if need_padding[2]:
                                w_in = w < X_sizes[2]
                            else:
                                w_in = True
                            if not is_zero(h_in * w_in):
                                pool.append([h_in * w_in * input.value[bi][k][h_in * h]
                                             [w_in * w], h_in, w_in, h, w])
                    process(pool, bi, k, i, j,operation.intermediate[3],output.grad,input.grad)

            
    prepare = get_prepare()
    if prepare:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if stride == None:
            stride = kernel_size
        padding = padding.upper() if isinstance(padding, str) else padding
        
        assert isinstance(input, Tensor)  ,"Invalid Input and weight"
        assert len(input.shape)==4,"Invalid Dimension input"
        if padding == 'SAME':
            output_shape = [int(math.ceil(shape[i] / strides[i])) for i in range(4)]
        else:
            output_shape = [input.shape[0],input.shape[1],(input.shape[2]-kernel_size[0])//stride[0]+1,
                            (input.shape[3]-kernel_size[1])//stride[1]+1 ]
             #out_shape.size:[Batch_size,out_channel,H_out,W_out]
        print_ln("%s",output_shape)
        new_value=MultiArray(output_shape,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        comparisons = MultiArray([input.shape[0],input.shape[1],
                                       output_shape[2], output_shape[3],
                                       kernel_size[0] * kernel_size[1]], sint)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,
                                  intermediate=[stride, kernel_size,padding,comparisons])
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,
                                  intermediate=[stride, kernel_size,padding,comparisons])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        strides=[1]+list(operation.intermediate[0])+[1]
        ksize=[1]+list(operation.intermediate[1])+[1]
        n_threads=8 if input.numel() > 2**20 else 1
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        training=input.req_grad
        batch=Array.create_from(regint.inc(N))
        def process(pool, bi, k, i, j,comparisons,Y,training):
            def m(a, b):
                c = a[0] > b[0]
                l = [c * x for x in a[1]]
                l += [(1 - c) * x for x in b[1]]
                return c.if_else(a[0], b[0]), l
            red = util.tree_reduce(m, [(x[0], [1] if training else [])
                                       for x in pool])
            Y[bi][k][i][j]= red[0]
            for ii, x in enumerate(red[1]):
                comparisons[bi][k][i][j][ii] = x

        Y_sizes =[N,output_h, output_w,n_channels_out]  
        X_sizes =[N,inputs_h, inputs_w,n_channels_in]
        need_padding = [strides[i] * (Y_sizes[i] - 1) + ksize[i] >
                        X_sizes[i] for i in range(4)]
        overlap = reduce(operator.or_, (x < y for x, y in zip(strides, ksize)) )
        @for_range_opt_multithread(n_threads,[len(batch), n_channels_in])
        def _(l, k):
            bi = batch[l]
            @for_range_opt(output_h)
            def _(i):
                h_base = strides[1] * i
                @for_range_opt(output_w)
                def _(j):
                    if overlap:
                        break_point()
                    w_base = strides[2] * j
                    pool = []
                    for ii in range(ksize[1]):
                        h = h_base + ii
                        if need_padding[1]:
                            h_in = h < X_sizes[1]
                        else:
                            h_in = True
                        for jj in range( ksize[2]):
                            w = w_base + jj
                            if need_padding[2]:
                                w_in = w < X_sizes[2]
                            else:
                                w_in = True
                            if not is_zero(h_in * w_in):
                                pool.append([h_in * w_in * input.value[bi][k][h_in * h]
                                             [w_in * w], h_in, w_in, h, w])
                    process(pool, bi, k, i, j,operation.intermediate[3],output.value,training)
        set_opid(op_id+1)  # record the input and output of the op
    return output

    
    


def avg_pool2d(input, kernel_size, stride=None, padding=0,):
    op_id = get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        n_threads=8 if input.numel() > 2**20 else 1
        pool_size=reduce(operator.mul, operation.intermediate[1])
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        strides=operation.intermediate[0]
        ksize=operation.intermediate[1]
        padding=operation.intermediate[2]
        batch=Array.create_from(regint.inc(N))
        if input.req_grad:
            get_tape().start_new_basicblock(name='')
            def process(pool, bi, k, i, j,nabla_Y,nabla_X,pool_size):
                part = nabla_Y[bi][k][i][j] * (1 / pool_size)
                for x, h_in, w_in, h, w in pool:
                    hh = h * h_in
                    ww = w * w_in
                    res = h_in * w_in * part
                    # get_program().protect_memory(True)
                    nabla_X[bi][k][hh][ww] += res
                    # get_program().protect_memory(False)
        Y_sizes = [N, output_h, output_w, n_channels_out]
        X_sizes = [N, inputs_h, inputs_w, n_channels_in]
        need_padding = [strides[i] * (Y_sizes[i] - 1) + ksize[i] >
                        X_sizes[i] for i in range(4)]
        @for_range_opt_multithread(n_threads, [N, n_channels_in])
        def _(l, k):
            bi = batch[l]
            @for_range_opt(Y_sizes[1])
            def _(i):
                h_base = strides[1] * i - padding[1]
                hs = [h_base + jj for jj in range(ksize[1])]
                if need_padding[1]:
                    h_ins = [(h < X_sizes[1]) * (h >= 0) for h in hs]
                else:
                    h_ins = [True] * ksize[1]

                @for_range_opt(Y_sizes[2])
                def _(j):
                    w_base = strides[2] * j - padding[1]
                    pool = []
                    ws = [w_base + jj for jj in range(ksize[2])]
                    if need_padding[2]:
                        w_ins = [(w < X_sizes[2]) * (w >= 0) for w in ws]
                    else:
                        w_ins = [True] * ksize[2]
                    for ii in range(ksize[1]):
                        h = hs[ii]
                        h_in = h_ins[ii]
                        for jj in range(ksize[2]):
                            w = ws[jj]
                            w_in = w_ins[jj]
                            if not is_zero(h_in * w_in):
                                pool.append([h_in * w_in * input.value[bi][k][h_in * h][w_in * w],
                                             h_in, w_in, h, w])
                    process(pool, bi, k, i, j, output.grad,input.grad,pool_size) 
    prepare = get_prepare()
    if prepare:
        if isinstance(kernel_size, int):
            kernel_size = (1,kernel_size, kernel_size,1)
        if isinstance(stride, int):
            stride = (1,stride, stride,1)
        if stride == None:
            stride = kernel_size
        padding = padding.upper() if isinstance(padding, str) else padding
        
        assert isinstance(input, Tensor)  ,"Invalid Input and weight"
        assert len(input.shape)==4,"Invalid Dimension input"

        if padding == 'SAME':
            output_shape = [int(math.ceil(input.shape[i] / strides[i])) for i in range(4)]
            padding = [0, 0]
        else:
            if padding == 'VALID':
                padding = 0
            if isinstance(padding, int):
                padding = [padding, padding]
            output_shape = [input.shape[0],input.shape[1]] + [
                (input.shape[2] + 2 * padding[0] - kernel_size[1]) //stride [1] + 1,
                (input.shape[3] + 2 * padding[1] - kernel_size[2]) //stride [2] + 1] 
             #out_shape.size:[Batch_size,H_out,W_out,out_channel]
             
        new_value=MultiArray(output_shape,input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,
                                  intermediate=[stride, kernel_size,padding])
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,
                                  intermediate=[stride, kernel_size,padding])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        strides=operation.intermediate[0]
        ksize=operation.intermediate[1]
        padding=operation.intermediate[2]
        n_threads=8 if input.numel() > 2**20 else 1
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        
        # assert n_channels_in == n_channels_out
        padding_h, padding_w = (0, 0)
        _,stride_h, stride_w,_ = operation.intermediate[0]
        _,filter_h, filter_w,_ = operation.intermediate[1]
        
        pool_size=reduce(operator.mul,operation.intermediate[1])
        
        batch=Array.create_from(regint.inc(N))
        def process(pool, bi, k, i, j,pool_size,Y):
            Y[bi][k][i][j] = sum(x[0] for x in pool) * (1 / pool_size)
        
        Y_sizes =[N,output_h, output_w,n_channels_out]  
        X_sizes =[N,inputs_h, inputs_w,n_channels_in]
        need_padding = [strides[i] * (Y_sizes[i] - 1) + ksize[i] >
                        X_sizes[i] for i in range(4)]
        @for_range_opt_multithread(n_threads,[N, n_channels_in])
        def _(l, k):
            bi = batch[l]
            @for_range_opt(Y_sizes[1])
            def _(i):
                h_base = strides[1] * i - padding[1]
                hs = [h_base + jj for jj in range(ksize[1])]
                if need_padding[1]:
                    h_ins = [(h < X_sizes[1]) * (h >= 0) for h in hs]
                else:
                    h_ins = [True] * ksize[1]
                @for_range_opt(Y_sizes[2])
                def _(j):
                    w_base = strides[2] * j - padding[1]
                    pool = []
                    ws = [w_base + jj for jj in range(ksize[2])]
                    if need_padding[2]:
                        w_ins = [(w < X_sizes[2]) * (w >= 0) for w in ws]
                    else:
                        w_ins = [True] * ksize[2]
                    for ii in range(ksize[1]):
                        h = hs[ii]
                        h_in = h_ins[ii]
                        for jj in range(ksize[2]):
                            w = ws[jj]
                            w_in = w_ins[jj]
                            if not is_zero(h_in * w_in):
                                pool.append([h_in * w_in * input.value[bi][k][h_in*h][w_in * w],
                                             h_in, w_in, h, w])
                    process(pool, bi, k, i, j,pool_size,output.value)
        set_opid(op_id+1)
    return output  


def dropout(input, p=0.5, training=False, inplace=False):  # todo
    op_id = get_opid()
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        bin_value, = operation.intermediate
        dl_dself = dl_d[operation.inputs[0]]
        dl_dself[:] += 1 / (1 - p) * bin_value[:] * dl_dx[:]
            
    prepare = get_prepare()
    if prepare:
        assert isinstance(input, Tensor), "Invalid Input"
        if isinstance(input.value,Array):
            new_value = Array(input.sizes[0], input.value.value_type)
            bin_value = Array(input.sizes[0], input.value.value_type)
        else:
            new_value = MultiArray(input.sizes, input.value.value_type)
            bin_value = MultiArray(input.sizes, input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate, intermediate=[bin_value])
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate, intermediate=[bin_value])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        bin_value, = operation.intermediate
        if training:
            n_bits = -math.log(p, 2)
            assert n_bits == int(n_bits)
            n_bits = int(n_bits)
            
            B = util.tree_reduce(util.or_op, 
                    (sint.get_random_bit(size=input.value.total_size())
                        for i in range(n_bits)))
            bin_value.assign_vector(B)
            
            output.value.assign_vector(1 / (1 - p) *
                input.value.get_vector() * B.get_vector())
        else:
            output.value[:] = input.value[:]
        set_opid(op_id+1)  # record the input and output of the op
    return output

#wqruan: seems useless
# def one_hot(input, num_classes=-1):
#     # i think user should specify the num_classes, if not, we should calculate the max value in input.
#     """example:
#     one_hot(torch.tensor([0, 1, 2, 3, 4]), num_classes=8)
#     tensor([[1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0]])"""
#     assert isinstance(input, Tensor), "input should be Tensor"
#     assert input.value.value_type == cint, "input should be cint"
#     x = input.value
#     in_sizes = x.sizes
#     b = reduce(operator.mul, in_sizes) if len(in_sizes) >= 2 else in_sizes[0]
#     output = MultiArray([*in_sizes, num_classes], x.value_type)

#     output.view(-1, num_classes)

#     for i in range(b):
#         output[i][x.get_vector()[i]] = 1

#     output.view(*in_sizes, num_classes)
#     return Tensor(output)


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):  # todo
    pass


# we should replace inv(std) to invsrqt(var) later
def batch_norm(input, running_mean, running_std, weight=None, bias=None, training=False, eps=1e-05, momentum=0.1):
    
    assert isinstance(input,Tensor) ,"Invalid input"
    
    new_sizes = [(input.value.sizes[i] if i == 1 else 1) for i in range(len(input.value.sizes))]
    if isinstance(running_mean.value, Array):
        running_mean.value = running_mean.value.reshape(new_sizes)
    if isinstance(running_std.value, Array):
        running_std.value = running_std.value.reshape(new_sizes)    
        
    if training:
        x_mean = input.mean(dim=[0,2,3], keepdim=True)
        x_std = input.std(dim=[0,2,3], keepdim=True) 
        running_mean = x_mean * momentum + running_mean * (1-momentum)
        running_std = x_std * momentum + running_std * (1-momentum)
    else:
        x_mean = running_mean
        x_std = running_std
        
    output = (input - x_mean) / (x_std + eps) 
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


# we should replace inv(std) to invsrqt(var) later
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    
    assert isinstance(input,Tensor) ,"Invalid input"
    
    dim = []
    for i in range(len(normalized_shape)):
        assert normalized_shape[len(normalized_shape)-1-i] == input.sizes[len(input.sizes)-1-i] ,"Invalid normalized_shape"
        dim.append(len(input.sizes)-1-i)
    dim.reverse()
    
    x_mean = input.mean(dim=dim, keepdim=True)
    x_std = input.std(dim=dim, keepdim=True) 
    
    output = (input - x_mean) / (x_std + eps) 
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


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


def mse_loss(input, target, reduction='mean'): # todo
    op_id = get_opid()
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        dl_dself = dl_d[operation.inputs[0]]
        
        dx = input.value[:] - target.value[:]
        dl_dself[:] += 2 * dx * dl_dx[:]
        
        if reduction == 'mean':
            dl_dself[:] /= input.value.total_size()
        
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    prepare = get_prepare()
    if prepare:
        new_value = Array(1, input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
    
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        set_opid(op_id+1)  # record the input and output of the op
    else:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        
        dx = input.value[:] - target.value[:]
        dx2 = dx * dx
        sumdx2 = sum(dx2)
        
        output.value[:] = sumdx2
        if reduction == 'mean':
            output.value[:] /= input.value.total_size()
        else:
            assert reduction == 'sum'
        set_opid(op_id+1)  # record the input and output of the op
    return output


def binary_cross_entropy(input, target, weight=None):
    pass


def cross_entropy(input, target, weight=None):
    pass
