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
approx = False


@buildingblock("relu-forward")
def relu(input, inplace=False):  
    # Considering that the saved memory overhead has very little impact on MPC computing performance, 
    #the inplace parameter is not considered
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-relu-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        dl_d[input.name]+=operation.intermediate[0][:]*dl_dy[:]        
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
            inter=Array(input.shape[0],sint)
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
            inter=MultiArray(list(input.shape) ,sint)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,intermediate=[inter])
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,intermediate=[inter])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        output = tensors[operation.outputs[0]]
        larger=0 < input.value[:]
        operation.intermediate[0].assign_vector(larger)
        if not forward:
            set_init_op_id(init_op_id+1)
        output.value[:] = (larger).if_else(input.value[:], 0) 
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

@buildingblock("sigmoid-forward")
def sigmoid(input,approx=False): # added approx parameter to speed up the computation
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-sigmoid-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        # if input_.req_grad:
        dl_d[input_.name]+=output.value[:]*(1-output.value[:])*dl_dy[:]
            
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
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
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if not forward:
            set_init_op_id(init_op_id+1)
        if approx:
            output.value[:]=approx_sigmoid(input.value[:])
        else:
            output.value[:] =  sigmoid_from_e_x(input.value[:],exp(-input.value[:]))
    set_opid(op_id+1)  # record the input and output of the op
    return output

@buildingblock("logsigmoid-forward")
def logsigmoid(input):  # todo
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-19]+"-logsigmoid-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        input_ = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if input_.req_grad:
            dl_d[input_.name]+=1/(1+exp(output.value[:]))*dl_dy[:]
            
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
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
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if not forward:
            set_init_op_id(init_op_id+1)
        output.value[:] = -log_e(1+exp(-input.value[:]))
    set_opid(op_id+1)  # record the input and output of the op
    return output

@buildingblock("tanh-forward")
def tanh(input):  # todo
    return input.tanh()
    # op_id = get_opid()
    # @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-tanh-backward")
    # def propagate(dl_doutputs, operation):
    #     dl_dy, = dl_doutputs
    #     input_ = tensors[operation.inputs[0]]
    #     output = tensors[operation.outputs[0]]
    #     dl_d[input_.name]+=(1-output.value[:]*output.value[:])*dl_dy[:]
            
    # prepare = get_prepare()
    # if prepare:
    #     assert isinstance(input, Tensor),"Invalid Input"
    #     if isinstance(input.value,Array):
    #         new_value=Array(input.shape[0],input.value.value_type)
    #     else:
    #         new_value=MultiArray(list(input.shape) ,input.value.value_type)
    #     output = Tensor(new_value, req_grad=input.req_grad)
    #     if input.req_grad:
    #         operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
    #     else:
    #         operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
    #     gradient_operation.append(operation)
    #     operation_id = len(gradient_operation) - 1
    #     op_id_store[op_id] = operation_id
    #     set_opid(op_id+1)
    # else:
    #     operation = gradient_operation[op_id_store[op_id]]
    #     input = tensors[operation.inputs[0]]
    #     output = tensors[operation.outputs[0]]
    #     x=input.value[:]
    #     ex=exp(x)
    #     e_x=exp(-x)
    #     output.value[:] = sanitize(x, (ex-e_x)/(ex+e_x), -1, 1)    
    #     set_opid(op_id+1)  # record the input and output of the op
    # return output
    

@buildingblock("softmax-forward")
def softmax(input,dim=-1):
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-softmax-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        output = tensors[operation.outputs[0]]
        output.value.print_reveal_nested()
        if isinstance(input.value, MultiArray):
            # dl_dx = softmax(x)*(   dl_dy    -    (dl_dy*softmax(x)).sum(dim=-1)  )
            inter_sum=operation.intermediate[2]
            inter_inital0=operation.intermediate[3]
            inter_broadcast_sub=operation.intermediate[4]
            dl_dy.element_wise_mul(output.value,inter_inital0 )
            inter_inital0.sum(dim,res=inter_sum,keepdims=True)
            boardcasted_multiarray_sub(dl_dy, inter_sum,inter_inital0)
            output.value.element_wise_mul(inter_inital0, inter_inital0)
            # print_ln('softmax backward:end:')
            # inter_inital0.print_reveal_nested()
            dl_d[operation.inputs[0]][:] += inter_inital0[:]
        else:
            res = output.value[:]*(dl_dy[:]-sum(output.value[:]*dl_dy[:]))
            dl_d[operation.inputs[0]][:] += res
            
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
            inter=[]
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
            changed_size=list(input.shape)
            changed_size=input.value.tuple_permute(input.shape,get_permute(len(input.sizes), [dim%len(input.sizes)])) #dim=2,input:[4,3,2,5]-->[4,3,5,2]
            inter=[MultiArray(changed_size,input.value.value_type),MultiArray(changed_size,input.value.value_type)]
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            if isinstance(input.value,MultiArray):
                reduced_dim=list(input.shape)
                reduced_dim[dim]=1
                inter_sum=MultiArray(reduced_dim,input.value.value_type)  
                dims, v1, _ = reconst_dims(output.value, inter_sum)
                target_size = v1.tuple_permute(output.value.sizes, get_permute(len(output.sizes), dims))        
                inter+=[inter_sum,MultiArray(list(input.shape) ,input.value.value_type)
                        ,MultiArray(target_size ,input.value.value_type)]       
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,intermediate=inter)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,intermediate=inter)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        if not forward:
            set_init_op_id(init_op_id+1)
        if isinstance(input.value,Array):
            output.value.assign_vector(vec_softmax(input.value.get_vector()),0)
        else:
            changed_0= operation.intermediate[0]  
            changed_output_1=operation.intermediate[1]
            input.value.permute_without_malloc( changed_0 ,get_permute(len(output.sizes), [(dim)%len(output.sizes)]))      
            times, num_per_time = reduce(operator.mul, changed_0.shape[:-1]) if len(changed_0.shape[:-1]) >= 1 else 1, changed_0.shape[-1]
            @for_range_opt(times)
            def _(i):
                changed_output_1.assign_vector(vec_softmax(changed_0.get_vector(i*num_per_time, num_per_time)), i*num_per_time)
            break_point()
            
            changed_output_1.permute_without_malloc(output.value,get_permute(len(output.sizes), [ dim%len(output.sizes) ]))
        
    set_opid(op_id+1)  # record the input and output of the op
    return output
    
def vec_softmax(x):
    max = util.max(x)
    index = x == max
    tmp = x*x*x
    tmp = x> tmp
    e_x = mpc_math.exp_fx(x -max , 8)
    sfix.all_pos = True
    res = e_x  / sum(e_x)
    sfix.all_pos = False
    return res


# def log_softmax(input, dim=-1):  # todo
#     tmp=softmax(input=input,dim=dim)
#     return tmp.log()


def log_softmax(input, dim=-1):  # todo
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-tanh-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
        softmax_value=operation.intermediate[0]
        if isinstance(input.value, MultiArray):
            # dl_dx =  dl_dy  -  softmax* sum( dl_dy , dim ) 
            inter_sum=operation.intermediate[4]
            inter_inital0=operation.intermediate[5]
            inter_broadcast_sub=operation.intermediate[6]
            dl_dy.sum(dim,res=inter_sum,keepdims=True)
            boardcasted_multiarray_mul(softmax_value,inter_sum,inter_inital0)
            dl_d[operation.inputs[0]][:] +=  (dl_dy[:]-inter_inital0[:])
        else:
            res = dl_dy[:]-(sum(dl_dy)*softmax_value[:])
            dl_d[operation.inputs[0]][:] += res     

    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert isinstance(input, Tensor),"Invalid Input"
        assert isinstance(dim, int) , "dim is not int"
        if isinstance(input.value,Array):
            new_value=Array(input.shape[0],input.value.value_type)
            inter=[Array(input.shape[0],input.value.value_type)]
        else:
            new_value=MultiArray(list(input.shape) ,input.value.value_type)
            changed_size=list(input.shape)
            changed_size=input.value.tuple_permute(input.shape,get_permute(len(input.sizes), [dim%len(input.sizes)])) #dim=2,input:[4,3,2,5]-->[4,3,5,2]
            inter=[MultiArray(list(input.shape) ,input.value.value_type),MultiArray(changed_size,input.value.value_type),
                   MultiArray(changed_size,input.value.value_type),MultiArray(changed_size,input.value.value_type)] 
            #softmax,changed_0,changed_output_0,changed_output_2
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            if isinstance(input.value,MultiArray):
                reduced_dim=list(input.shape)
                reduced_dim[dim]=1
                inter_sum=MultiArray(reduced_dim,input.value.value_type)  
                dims, v1, _ = reconst_dims(output.value, inter_sum)
                target_size = v1.tuple_permute(output.value.sizes, get_permute(len(output.sizes), dims))      
                inter+=[inter_sum,MultiArray(list(input.shape) ,input.value.value_type)
                        ,MultiArray(target_size ,input.value.value_type)]       
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,intermediate=inter)
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,intermediate=inter)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        softmax_value=operation.intermediate[0]
        if not forward:
            set_init_op_id(init_op_id+1)        
        if isinstance(input.value,Array):
            logsoftmax_sfix,softmax_sfix=vec_logsoftmax_softmax(input.value.get_vector())
            output.value.assign_vector(logsoftmax_sfix,0)
            softmax_value.assign_vector(softmax_sfix)
        else:
            changed_0= operation.intermediate[1]  #store permuted input value
            changed_output_1=operation.intermediate[2] #store permuted logsoftmax
            changed_output_2=operation.intermediate[3] #store permuted softmax
   
            input.value.permute_without_malloc( changed_0 ,get_permute(len(output.sizes), [dim%len(output.sizes)]))      
            times, num_per_time = reduce(operator.mul, changed_0.shape[:-1]) if len(changed_0.shape[:-1]) >= 1 else 1, changed_0.shape[-1]
            @for_range_opt(times)
            def _(i):
                logsoftmax_sfix,softmax_sfix=vec_logsoftmax_softmax(changed_0.get_vector(i*num_per_time, num_per_time))
                changed_output_1.assign_vector(logsoftmax_sfix, i*num_per_time)
                changed_output_2.assign_vector(softmax_sfix, i*num_per_time)
            break_point()
            
            changed_output_1.permute_without_malloc(output.value,get_permute(len(output.sizes), [dim%len(output.sizes)]))
            changed_output_2.permute_without_malloc(operation.intermediate[0],get_permute(len(output.sizes), [dim%len(output.sizes)]))
        
    set_opid(op_id+1)  # record the input and output of the op
    return output
 
def vec_logsoftmax_softmax(x):
    x_minus_max=x - util.max(x)
    e_x = mpc_math.exp_fx(x_minus_max)
    sumex=sum(e_x)
    logsum=mpc_math.log_fx(sumex,math.e)
    return x_minus_max-logsum , e_x / sumex


@buildingblock("linear")
def linear(input, weight, bias=None):
    assert isinstance(input,Tensor),"Invalid input"
    assert isinstance(weight,Tensor),"Invalid weight"
    assert input.shape[-1]==weight.shape[-1],"Invalid Dimension"
    if len(input.sizes) > len(weight.sizes):
        output=input.single_bmm(weight.transpose())
    elif len(input.sizes) == len(weight.sizes):
        output=input.mm(weight.transpose())
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

@buildingblock("conv2d-forward")
def conv2d(input:Tensor, weight:Tensor, bias=None, stride=[1,1], padding=[0,0], groups = 1):
    #input.shape:(batch_size,channel_in,H,W)
    #weight.shape:(out_channels, in_channels // groups, H,W)
    #bais:(out_channels)
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-15]+"-conv2d-backward")
    def propagate(dl_doutputs, operation):
        # dl_dy, = dl_doutputs
        input = tensors[operation.inputs[0]]
        weight= tensors[operation.inputs[1]] # N C/G H W
        output = tensors[operation.outputs[0]] 
        _, _,weights_h, weights_w= weight.shape
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape
        input_value=input.value.permute([0,2,3,1])
        weight_value=weight.value.permute([0,2,3,1]) 
        nabla_Y=output.grad.permute([0,2,3,1])

        stride_h, stride_w = stride
        padding_h, padding_w = padding
        print("padding:",padding)
        
        n_threads=8 if input.numel() > 2**20 else 1
        batch=Array.create_from(regint.inc(N))
        input_size = inputs_h * inputs_w * N #why have no channel_in? 128*36
        batch_repeat = regint.Matrix(N, inputs_h * inputs_w) # 128,6*6
        batch_repeat.assign_vector( batch.get(
            regint.inc(input_size, 0, 1, 1, N)) * reduce(operator.mul, input_value.sizes[1:]) )
        @for_range_opt_multithread(n_threads, [int(n_channels_in/groups), n_channels_out])
        def _(i, j):
            a = regint.inc(input_size, input_value.address + i + j//int(n_channels_out/groups)*int(n_channels_out/groups), n_channels_in, N,
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
            weight.grad.assign_vector_by_indices(reduced, j, i, None, None)  
        
        #TODO : Support groups backward for X
        # print("\nbackward for X start:\n")
        # print(weights_h,weights_w,inputs_h, inputs_w,output_h, output_w)
        
        # nabla_X=input.grad.permute([0,2,3,1])
        # reverse_weights = MultiArray(
        #         [n_channels_in, weights_h, weights_w, n_channels_out], sfix)
        # @for_range_opt_multithread(n_threads, n_channels_in)
        # def _(l):
        #     @for_range(weights_h)
        #     def _(j):
        #         @for_range(weights_w)
        #         def _(k):
        #             addresses = regint.inc(n_channels_out,
        #                 weight_value[0][j][weights_w-k-1].get_address(l),
        #                 reduce(operator.mul, weight_value.sizes[1:]))
        #             reverse_weights[l][weights_h-j-1][k].assign_vector(
        #                 weight_value.value_type.load_mem(addresses))
        # padded_w = inputs_w + 2 * padding_w
        # padded_h = inputs_h + 2 * padding_h
        # if padding_h or padding_w:
        #     output = MultiArray(
        #         [N, padded_h, padded_w, n_channels_in], sfix)
        # else:
        #     output = nabla_X
        # @for_range_opt_multithread(n_threads,
        #                             [N, n_channels_in])
        # def _(i, j):
        #     res = sint(size = (padded_w * padded_h))
        #     conv2ds(res, nabla_Y[i].get_vector().v,
        #             reverse_weights[j].get_vector().v,
        #             padded_h, padded_w, output_h, output_w,
        #             weights_h, weights_w, 1, 1, n_channels_out,
        #             weights_h - 1, weights_w - 1, 1)
        #     output.assign_vector_by_indices(
        #         unreduced_sfix._new(res).reduce_after_mul(),i, None, None, j)
            
        # if padding_h or padding_w:
        #     @for_range_opt_multithread(n_threads, N)
        #     def _(i):
        #         @for_range(inputs_h)
        #         def _(j):
        #             @for_range(inputs_w)
        #             def _(k):
        #                 jj = j + padding_w
        #                 kk = k + padding_w
        #                 nabla_X[i][j][k].assign_vector(
        #                         output[i][jj][kk].get_vector())
        # nabla_X.permute_without_malloc(input.grad,[0,3,1,2])
        
        
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
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
        # set_opid(op_id+1)
    if not prepare or not forward:
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        operation = gradient_operation[op_id_store[op_id]]
        output= tensors[operation.outputs[0]] 
        _, _,weights_h, weights_w= weight.shape
        N,  n_channels_in,inputs_h, inputs_w = input.shape
        _,  n_channels_out,output_h, output_w = output.shape #B C H W
        input.value.view(N, groups, int(n_channels_in/groups), inputs_h, inputs_w) # N G C/G H W
        input_value = MultiArray([groups, N, inputs_h, inputs_w, int(n_channels_in/groups)], input.value.value_type) # G B H W C/G
        input.value.permute_without_malloc(input_value, [1,0,3,4,2]) # # G B H W C/G
        weight_value = MultiArray([weight.sizes[0], weight.sizes[2], weight.sizes[3], weight.sizes[1]], weight.value.value_type)
        weight.value.permute_without_malloc(weight_value, [0,2,3,1]) # N, H, W, C/G
        output_value = MultiArray([output.sizes[0], output.sizes[2], output.sizes[3], output.sizes[1]], output.value.value_type)
        output.value.permute_without_malloc(output_value, [0,2,3,1]) # B, H, W, N
        n_channels_in_group = int(n_channels_in / groups)
        n_threads=8 if input.numel() > 2**20 else 1
        if not forward:
            set_init_op_id(init_op_id+1)   
        n_parts = 1
        while N % n_parts != 0:
            n_parts -= 1
        # print('Convolution in %d parts' % n_parts)
        unreduced = MultiArray(output_value.sizes, sint, address=output_value.address)
        part_size =N // n_parts
        size_=part_size*reduce(operator.mul,input.value.sizes[2:])
        @for_range_multithread(n_threads, 1, [groups, int(n_channels_out/groups)]) #N
        def _(i, j):
            inputs = input_value.get_vector(i*size_,size_).v # B H W C/G
            weights = weight_value.get_part_vector(i*int(n_channels_out/groups)+j).v # N WH WW C/G
            res = sint(size = output_h * output_w * part_size) # B, OUT_W, OUT_H
            conv2ds(res, inputs, weights, output_h, output_w,
                    inputs_h, inputs_w, weights_h, weights_w,
                    stride_h, stride_w, n_channels_in_group, padding_h, padding_w,
                    part_size)
            if bias:
                res += bias.value.expand_to_vector(i*int(n_channels_out/groups)+j, res.size).v
            addresses = regint.inc(res.size,
                                    unreduced[0].address + i*int(n_channels_out/groups)+j,
                                    n_channels_out)
            res.store_in_mem(addresses)
            
        n_outputs = N * reduce(operator.mul, output_value.sizes[1:])
        @multithread(n_threads, n_outputs,
                     1000 if sfix.round_nearest else 10 ** 6)                                                                                
        def _(base, n_per_thread):
            res = sfix().unreduced(sint.load_mem(unreduced.address + base,
                              size=n_per_thread),sfix).reduce_after_mul()
            
            res.store_in_mem(output_value.address + base) #B H W N
        output_value.permute_without_malloc(output.value, [0,3,1,2])
        input.value.view(N, n_channels_in, inputs_h, inputs_w)
        input_value.delete()
        output_value.delete()
        weight_value.delete()
        
    set_opid(op_id+1)  # record the input and output of the op
    return output


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, outputpadding=0):
     pass

@buildingblock("max_pool2d-forward")
def max_pool2d(input, kernel_size=2, stride=2, padding=0):
    op_id=get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-19]+"-max_pool2d-backward")
    def propagate(dl_doutputs, operation):
        dl_dy, = dl_doutputs
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
    init_op_id = get_init_op_id()
    forward = get_forward()
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
            # if isinstance(stride, int):
            strides = (1, 1, stride[0], stride[0])
            output_shape = [int(math.ceil(input.shape[i] / strides[i])) for i in range(4)]
        else:
            output_shape = [input.shape[0],input.shape[1],(input.shape[2]-kernel_size[0])//stride[0]+1,
                            (input.shape[3]-kernel_size[1])//stride[1]+1 ]
             #out_shape.size:[Batch_size,out_channel,H_out,W_out]
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
        # set_opid(op_id+1)
    if not prepare or not forward:
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
        if not forward:
            set_init_op_id(init_op_id+1)
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

    
    

@buildingblock("avg_pool2d-forward")
def avg_pool2d(input, kernel_size, stride=None, padding=0,):
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-19]+"-avg_pool2d-backward")
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
    init_op_id = get_init_op_id()
    forward = get_forward()
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
        # set_opid(op_id+1)
    if not prepare or not forward:
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
        if not forward:
            set_init_op_id(init_op_id+1)        
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

@buildingblock("dropout-forward")
def dropout(input, p=0.5, training=False, inplace=False):  # todo
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-dropout-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        bin_value, = operation.intermediate
        dl_dself = dl_d[operation.inputs[0]]
        dl_dself[:] += 1 / (1 - p) * bin_value[:] * dl_dx[:]
            
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
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
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        bin_value, = operation.intermediate
        if not forward:
            set_init_op_id(init_op_id+1)
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

@buildingblock("normalize")
def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    assert p == 2  # todo
    assert isinstance(dim, (int, list))
    if isinstance(dim, int):
        dim = [dim]
        
    xp = input * input
    xpsum = xp.sum(dim=dim, keepdim=True)
    xpsumSqr = xpsum.invsqrt(eps=eps)
    return input * xpsumSqr
    

@buildingblock("batch_norm")
def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, eps=1e-05, momentum=0.1):
    
    assert isinstance(input,Tensor) ,"Invalid input"
    # assert input.value.sizes[1] == running_mean.value.sizes[1], "Invalid input"
    # assert input.value.sizes[1] == running_var.value.sizes[1], "Invalid input"
    # assert input.value.sizes[1] == weight.value.sizes[1], "Invalid input"
    # assert input.value.sizes[1] == bias.value.sizes[1], "Invalid input"
    
    new_sizes = [(input.value.sizes[i] if i == 1 else 1) for i in range(len(input.value.sizes))]
    if isinstance(running_mean.value, Array):
        running_mean.value = running_mean.value.reshape(new_sizes)
    if isinstance(running_var.value, Array):
        running_var.value = running_var.value.reshape(new_sizes)
    if isinstance(weight.value, Array):
        weight.value = weight.value.reshape(new_sizes)
        weight.grad = weight.grad.reshape(new_sizes)
    if isinstance(bias.value, Array):
        bias.value = bias.value.reshape(new_sizes)
        bias.grad = bias.grad.reshape(new_sizes)
    
    if training:
        x_mean = input.mean(dim=[0,2,3], keepdim=True)
        # x_var = input.std(dim=[0,2,3], keepdim=True) 
        x_var = input.var(dim=[0,2,3], keepdim=True, unbiased=True) #5s
        running_mean.value[:] = x_mean.value[:] * momentum + running_mean.value[:] * (1-momentum)
        running_var.value[:] = x_var.value[:] * momentum + running_var.value[:] * (1-momentum)
    else:
        x_mean = running_mean
        x_var = running_var
    x_var = x_var + eps # todo
    output = (input - x_mean) * x_var.invsqrt() #9s 5s 4s
    # output = (input - x_mean) / x_var
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output


@buildingblock("layer_norm")
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    
    assert isinstance(input,Tensor) ,"Invalid input"
      
    dim = []
    for i in range(len(normalized_shape)):
        assert normalized_shape[len(normalized_shape)-1-i] == input.sizes[len(input.sizes)-1-i] ,"Invalid normalized_shape"
        dim.append(len(input.sizes)-1-i)
    dim.reverse()
    
    new_sizes = [(input.value.sizes[i] if i in dim else 1) for i in range(len(input.value.sizes))]
    if isinstance(weight.value, Array):
        weight.value = weight.value.reshape(new_sizes)
        weight.grad = weight.grad.reshape(new_sizes)
    if isinstance(bias.value, Array):
        bias.value = bias.value.reshape(new_sizes)
        bias.grad = bias.grad.reshape(new_sizes)
    
    x_mean = input.mean(dim=dim, keepdim=True)
    x_var = input.var(dim=dim, keepdim=True, unbiased=True) 
    
    x_var = x_var + eps
    output = (input - x_mean) * x_var.invsqrt() 

    if weight is not None:
        output = output * weight

    if bias is not None:
        output = output + bias
    return output


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    assert isinstance(dim, int)
    dim = [dim]

    x1_ = normalize(x1, 2, dim, eps)
    x2_ = normalize(x2, 2, dim, eps)
    xx = x1_ * x2_
    return xx.sum(dim=dim, keepdim=False)


def pdist(input, p=2):  # todo
    pass

@buildingblock("kl_div-forward")
def kl_div(input, target, log_target=False,reduction='mean'):
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-15]+"-kl_div-backward")
    def propagate(dl_doutputs, operation):
        input=tensors[operation.inputs[0]]
        inter=operation.intermediate
        if inter[-1]=='mean':
            dl_d[input.name][:]+=(-1/input.numel())*inter[0][:]
        elif inter[-1]=='batchmean':
            dl_d[input.name][:]+=(-1/input.sizes[0])*inter[0][:]
        else:
            dl_d[input.name][:]-=inter[0][:]
        
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert isinstance(input, Tensor) and isinstance(target, Tensor), "Invalid Input"
        assert len(input.sizes)==len(target.sizes),"Inequal dimension"
        assert reduction in ['mean','sum','batchmean'],"invalid reduction"
        if isinstance(input.value,Array):
            inter = Array(input.value.length, input.value.value_type)
        else:
            inter = MultiArray(input.value.sizes, input.value.value_type)
        new_value=Array(1, input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name,target.name], outputs=[output.name], propagate=propagate, intermediate=[inter,reduction])
        else:
            operation = Operation(inputs=[input.name,target.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter,reduction])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        target= tensors[operation.inputs[1]]
        output = tensors[operation.outputs[0]]
        res=0
        if not forward:
            set_init_op_id(init_op_id+1)
        if log_target:
            t=mpc_math.pow_fx(math.e,target.value[:])
            operation.intermediate[0].assign_vector(t)
            tmp=t*(target.value[:]-input.value[:])
            res=sum(tmp)
        else:
            tmp=mpc_math.log_fx(target.value[:],math.e)
            operation.intermediate[0].assign_vector(target.value[:])
            res=sum(target.value[:]*(tmp-input.value[:]))
        if reduction=='mean':
            output.value[0]=res/input.numel()
        elif reduction=='batchmean':
            output.value[0]=res/input.sizes[0]
        else:
            output.value[0]=res
    set_opid(op_id+1)  # record the input and output of the op
    return output



@buildingblock("l1_loss-forward")
def l1_loss(input, target,reduction='mean'):
    op_id = get_opid()
    @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-l1_loss-backward")
    def propagate(dl_doutputs, operation):
        input=tensors[operation.inputs[0]]
        if operation.intermediate[-1]=='mean':
            dl_d[operation.inputs[0]][:]+= (operation.intermediate[0][:]/input.numel())
        else:
            dl_d[operation.inputs[0]][:]+= operation.intermediate[0][:]
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert isinstance(input, Tensor) and isinstance(target, Tensor), "Invalid Input"
        assert len(input.sizes)==len(target.sizes),"Inequal dimension"
        assert reduction in ['mean','sum'],"invalid reduction"
        if isinstance(input.value,Array):
            inter = Array(input.value.length, input.value.value_type)
        else:
            inter = MultiArray(input.value.sizes, input.value.value_type)
        new_value=Array(1, input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if input.req_grad:
            operation = Operation(inputs=[input.name,target.name], outputs=[output.name], propagate=propagate, intermediate=[inter,reduction])
        else:
            operation = Operation(inputs=[input.name,target.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter,reduction])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        target= tensors[operation.inputs[1]]
        output = tensors[operation.outputs[0]]
        if not forward:
            set_init_op_id(init_op_id+1)
        larger = input.value[:]>target.value[:]
        less=input.value[:]<target.value[:]
        final=larger-less
        operation.intermediate[0].assign_vector(final)
        total=input.numel()
        Sum= sum( final * (input.value[:]-target.value[:]))
        if reduction=='sum':
            output.value[0]=Sum
        elif reduction=='mean' : #mean
            output.value[0]=Sum/total
    set_opid(op_id+1)  # record the input and output of the op
    return output


@buildingblock("nll_loss-forward")
def nll_loss(input, target, weight=None,reduction='mean'):
    op_id = get_opid()
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-17]+"-nll_loss-backward")
    def propagate(dl_doutputs, operation):
        if reduction=='mean':
            dl_d[input.name].assign_vector( ( inter[:] ) /input.sizes[0] )
        else:
            dl_d[input.name].assign_vector(inter[:] )
    # forward
    prepare = get_prepare()
    init_op_id = get_init_op_id()
    forward = get_forward()
    if prepare:
        assert target.sizes==input.sizes,"Dimension invalid"
        new_value = Array(1, input.value.value_type)
        output = Tensor(new_value, req_grad=input.req_grad)
        if isinstance(input.value,Array):
            inter = Array(input.value.length, input.value.value_type)
        else:
            inter = MultiArray(input.value.sizes, input.value.value_type)
    
        if input.req_grad:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate,intermediate=[inter])
        else:
            operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate,intermediate=[inter])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # set_opid(op_id+1)
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        leq=input.value[:]>=0
        tmp=(2*leq-1)*target.value[:]
        output.value[:]=sum( input.value[:]*tmp)
        operation.intermediate[0].assign_vector(tmp)
        if not forward:
            set_init_op_id(init_op_id+1)
        if reduction == 'mean':
            output.value[:] *= 1 / input.sizes[0]
        else:
            assert reduction == 'sum' , 'reduction should be mean or sum'
    set_opid(op_id+1)  # record the input and output of the op
    return output

    


@buildingblock("mse_loss-forward")
def mse_loss(input, target, reduction='mean'):
    # op_id = get_opid()
    # backward
    # @backwardbuildingblock(get_program().globalbuildingblock[:-17]+"-mse_loss-backward")
    # def propagate(dl_doutputs, operation):
    #     dl_dx, = dl_doutputs
    #     dl_dself = dl_d[operation.inputs[0]]
        
    #     dx = input.value[:] - target.value[:]
    #     dl_dself[:] += 2 * dx * dl_dx[:]
        
    #     if reduction == 'mean':
    #         dl_dself[:] /= input.value.total_size()
        
    #     dl_dinputs = [dl_dself]
    #     return dl_dinputs
    # # forward
    # prepare = get_prepare()
    # if prepare:
    #     new_value = Array(1, input.value.value_type)
    #     output = Tensor(new_value, req_grad=input.req_grad)
    
    #     if input.req_grad:
    #         operation = Operation(inputs=[input.name], outputs=[output.name], propagate=propagate)
    #     else:
    #         operation = Operation(inputs=[input.name], outputs=[output.name], propagate=fake_propagate)
    #     gradient_operation.append(operation)
    #     operation_id = len(gradient_operation) - 1
    #     op_id_store[op_id] = operation_id
    #     set_opid(op_id+1)  # record the input and output of the op
    # else:
    #     operation = gradient_operation[op_id_store[op_id]]
    #     input = tensors[operation.inputs[0]]
    #     output = tensors[operation.outputs[0]]
    #     dx = input.value[:] - target.value[:]
    #     dx2 = dx * dx
    #     sumdx2 = sum(dx2)
        
    #     output.value[:] = sumdx2
    #     if reduction == 'mean':
    #         print(type(input.value.total_size()))
    #         output.value[:] *= 1 / input.value.total_size()
    #     else:
    #         assert reduction == 'sum' , 'reduction should be mean or sum'
    #     set_opid(op_id+1)  # record the input and output of the op
    # return output
    assert reduction == 'sum' or 'mean', 'reduction should be mean or sum'
    
    dx = input - target
    dx2 = dx * dx
    out = dx2.sum()
   
    if reduction == 'mean':
        out /= input.value.total_size()
    return out




def binary_cross_entropy(input, target, weight=None):
    pass


def cross_entropy(input, target, weight=None, reduction = 'mean'):
    tmp=log_softmax(input)
    return nll_loss(tmp,target,weight)

@buildingblock("gelu")
def gelu(input, approximate='tanh'):
    assert approximate == 'tanh' or 'Hardtanh', 'approximate of gelu should be tanh or Hardtanh'
    factor = input + input * input * input * 0.044715
    factor *= np.sqrt(2.0/np.pi)
    if approximate == 'tanh':
        factor = factor.tanh()
    else:
        factpr = factor.Hardtanh()
    factor += 1
    return factor * input * 0.5