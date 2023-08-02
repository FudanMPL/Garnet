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
from Compiler.instructions import *
from Compiler.instructions_base import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
# from Compiler.GC.types import sbitintis_train
from functools import reduce
from typing import List, NamedTuple, Callable, Dict, Optional, Union, Tuple, Any

_name = 1


class Operation(NamedTuple):
    inputs: List[str]
    outputs: List[str]

    # apply chain rule
    propagate: 'Callable[List[Tensor], List[Tensor]]'
    # forward : 'Callable[List[Tensor], List[Tensor]]'
    intermediate: List = []


# sotre of tensors involved in computation process
tensors = {}
# store of operators invovled in computation process, these operators are topologically sotred
gradient_operation: List[Operation] = []
# sotre of gradients involved in computation process, their types are tensors without gradient
dl_d = {}
# the flag indicates whether initialize gradients for tensors
is_train = True
# the flag indicated that we are in prepration phase, i.e. initialize inputs and outputs for each operators
global prepare
prepare = True
# op_id is used for extracting the references of inputs and outputs of one opertator
op_id = 0
# op_id_store stores the correlation among op_ids and operation ids.
op_id_store = {}

def fake_propagate(dl_doutputs, operation):
    pass

def check_boardcast_size(size1, size2):
    if len(size1)<len(size2):
        size1, size2 = size2, size1
    flag = 0
    for i in range(1, len(size2)+1):
        if size1[-i]!=size2[-i]:
            if size2[-i] == 1:
                flag = 1
            else:
                return False
        else:
            if flag == 1:
                return False
    return True
            
# As A * boardcast(B), the matrix is A here need reconstruct, 
# the new_matrix is a prepared memory for the result of reconstructed A and 
# the new_matrix is a [r, c] MultiArray which c is the total_size of B and
# r is the division of the total_size of A and B 
def matrix_reconst(matrix, new_matrix):
    r, c = new_matrix.sizes
    # new_matrix = MultiArray([r, c], matrix.value_type)
    # for i in range(0, r):
    #     for j in range(0, c):
    @for_range(r)
    def _(i):
        @for_range(c)
        def _(j):
            v = matrix.get_vector(j*r+i, 1)
            new_matrix.assign_vector(v, i*c+j)
    return new_matrix

def element_wise_add(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        temp_matrix = operation.intermediate[0]
        dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        if dl_dself.total_size()<dl_dother.total_size():
            v1, v2 = v2, v1
            req_grad1, req_grad2 = req_grad2, req_grad1
        # v1 back directly 
        if req_grad1:
            v1[:] += dl_dx[:]
        # broadcasted v2 back with reduce
        if req_grad2:
            dl_dx_rec = matrix_reconst(dl_dx, temp_matrix)
            # for i in range(0, v2.total_size()):
            @for_range(v2.total_size())
            def _(i):
                vsum = sum(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                v2.assign_vector(vsum, i) 
        
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        # check shape
        assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
        temp_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            else:
                new_value = MultiArray(other.value.sizes, self.value.value_type)
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], self.value.value_type)
            else:
                new_value = Array(other.value.sizes[0], self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        # check whether require grad
        if self.req_grad or other.req_grad:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate, intermediate=[temp_matrix])
        else:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate, intermediate=[temp_matrix])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]

        # swap to ensure v1 size is bigger than v2 size  
        v1, v2= input1.value, input2.value
        if input1.value.total_size() < input2.value.total_size():
            v1, v2 = v2, v1

        len1, len2 = v1.total_size(), v2.total_size()
        assert len1 % len2==0, "Invalid Dimension"
        # for i in range(0, len1//len2):
        #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        #     output.value.assign_vector(v3, i*len2)
        @for_range(len1//len2)
        def _(i):
            v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
            output.value.assign_vector(v3, i*len2)
        op_id += 1# record the input and output of the op
    return output


def element_wise_sub(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        temp_matrix = operation.intermediate[0]
        dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        # v1 back directly 
        if req_grad1:
            if dl_dself.total_size()<dl_dother.total_size():
                dl_dx_rec = matrix_reconst(dl_dx, v1.total_size())
                @for_range(v1.total_size())
                def _(i):
                    vsum = sum(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                    v1.assign_vector(vsum, i)                 
            else:
                v1[:] += dl_dx[:]
        # broadcasted v2 back with reduce
        if req_grad2:
            if dl_dself.total_size()<dl_dother.total_size():
                v2[:] += -1 * dl_dx[:]         
            else:
                dl_dx_rec = dl_dx_rec = matrix_reconst(dl_dx, temp_matrix)
                @for_range(v2.total_size())
                def _(i):
                    vsum = -1 * sum(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                    v2.assign_vector(vsum, i)     

        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        # check shape
        assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
        temp_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            else:
                new_value = MultiArray(other.value.sizes, self.value.value_type)
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], self.value.value_type)
            else:
                new_value = Array(other.value.sizes[0], self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        # check whether require grad
        if self.req_grad or other.req_grad:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate, intermediate=[temp_matrix])
        else:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate, intermediate=[temp_matrix])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]

        # swap to ensure v1 size is bigger than v2 size  
        v1, v2= input1.value, input2.value
        # if input1.value.total_size() < input2.value.total_size():
        #     v1, v2 = v2, v1

        len1, len2 = v1.total_size(), v2.total_size()
        assert len1 % len2==0, "Invalid Dimension"
        # for i in range(0, len1//len2):
        #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        #     output.value.assign_vector(v3, i*len2)
        if input1.value.total_size() > input2.value.total_size():
            @for_range(len1//len2)
            def _(i):
                v3 = v1.get_vector(i*len2, len2) - v2.get_vector(0, len2)
                output.value.assign_vector(v3, i*len2)
        else:
            @for_range(len2//len1)
            def _(i):
                v3 = v1.get_vector(0, len1) - v2.get_vector(i*len1, len1)
                output.value.assign_vector(v3, i*len1)  
        op_id += 1# record the input and output of the op
    return output


def element_wise_mul(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        temp1_matrix = operation.intermediate[0]
        temp2_matrix = operation.intermediate[1]
        dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        input1=tensors[operation.inputs[0]]
        input2=tensors[operation.inputs[1]]
        len1, len2 = v1.total_size(), v2.total_size()
        if dl_dself.total_size()<dl_dother.total_size():
            v1, v2 = v2, v1
            req_grad1, req_grad2 = req_grad2, req_grad1
            input1, input2 = input2, input1
        # v1 back directly 
        if req_grad1:
            @for_range_opt(len1//len2)
            def _(i):
                v3 = dl_dx.get_vector(i*len2, len2) * input2.value.get_vector(0, len2)
                v1.assign_vector(v3, i*len2)           
        # broadcasted v2 back with reduce
        if req_grad2:
            dl_dx_rec = matrix_reconst(dl_dx, temp1_matrix)
            input1_rec = matrix_reconst(input1.value, temp2_matrix)
            # for i in range(0, v2.total_size()):
            @for_range_opt(v2.total_size())
            def _(i):
                v3 = dl_dx.value_type.dot_product(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]), input1_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                v2.assign_vector(v3, i)    
        
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        # check shape
        assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
        temp1_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        temp2_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            else:
                new_value = MultiArray(other.value.sizes, self.value.value_type)
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], self.value.value_type)
            else:
                new_value = Array(other.value.sizes[0], self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        # check whether require grad
        if self.req_grad or other.req_grad:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate, intermediate=[temp1_matrix, temp2_matrix])
        else:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate, intermediate=[temp1_matrix, temp2_matrix])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]

        # swap to ensure v1 size is bigger than v2 size  
        v1, v2= input1.value, input2.value
        if input1.value.total_size() < input2.value.total_size():
            v1, v2 = v2, v1

        len1, len2 = v1.total_size(), v2.total_size()
        assert len1 % len2==0, "Invalid Dimension"
        # for i in range(0, len1//len2):
        #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        #     output.value.assign_vector(v3, i*len2)
        @for_range_opt(len1//len2)
        def _(i):
            v3 = v1.get_vector(i*len2, len2) * v2.get_vector(0, len2)
            output.value.assign_vector(v3, i*len2)
        op_id += 1# record the input and output of the op
    return output

def element_wise_div(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        inter = operation.intermediate[0]  # reuse the intervalue in mem
        temp1_matrix = operation.intermediate[1]
        temp2_matrix = operation.intermediate[2]
        dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        input1=tensors[operation.inputs[0]]
        input2=tensors[operation.inputs[1]]
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        len1, len2 = v1.total_size(), v2.total_size()
        # v1 back directly 
        if req_grad1:
            if dl_dself.total_size()<dl_dother.total_size():
                dl_dx_rec = matrix_reconst(dl_dx, temp1_matrix)
                inter_res = matrix_reconst(inter, temp2_matrix)
                @for_range_opt(v1.total_size())
                def _(i):
                    v3 = dl_dx.value_type.dot_product(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]), inter_res.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                    v1.assign_vector(v3, i)                  
            else:
                @for_range_opt(len1//len2)
                def _(i):
                    v3 = dl_dx.get_vector(i*len2, len2) * inter.get_vector(0, len2)
                    v1.assign_vector(v3, i*len2)  
        # broadcasted v2 back with reduce
        if req_grad2:
            if dl_dself.total_size()<dl_dother.total_size():
                v2 = -1 * inter[:] * inter[:] * dl_dx[:]
                @for_range_opt(len2//len1)
                def _(i):
                    v3 = v2.get_vector(i*len1, len1) * input1.get_vector(0, len1)
                    v2.assign_vector(v3, i*len1)                       
            else:
                dl_dx_rec = matrix_reconst(dl_dx, temp1_matrix)
                input1_rec = matrix_reconst(input1.value, temp2_matrix)   
                tmp_inter = -1 * inter[:] * inter[:]
                @for_range_opt(v2.total_size())
                def _(i):
                    v3 = dl_dx.value_type.dot_product(dl_dx_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]), input1_rec.get_vector(i*dl_dx_rec.sizes[1], dl_dx_rec.sizes[1]))
                    v2.assign_vector(v3, i)  
                v2[:] = v2[:] * tmp_inter

        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        # check shape
        assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
        temp1_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        temp2_matrix = MultiArray([other.value.total_size(), self.value.total_size()//other.value.total_size()], self.value.value_type)
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            else:
                new_value = MultiArray(other.value.sizes, self.value.value_type)
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], self.value.value_type)
            else:
                new_value = Array(other.value.sizes[0], self.value.value_type)
        inter = other.value.same_shape()
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        # check whether require grad
        if self.req_grad or other.req_grad:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate, intermediate=[inter, temp1_matrix, temp2_matrix])
        else:
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter, temp1_matrix, temp2_matrix])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]

        # swap to ensure v1 size is bigger than v2 size  
        v1, v2= input1.value, input2.value
        # if input1.value.total_size() < input2.value.total_size():
        #     v1, v2 = v2, v1

        len1, len2 = v1.total_size(), v2.total_size()
        assert len1 % len2==0, "Invalid Dimension"
        # for i in range(0, len1//len2):
        #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        #     output.value.assign_vector(v3, i*len2)
        tmp_inter = 1 / input2.value[:]
        operation.intermediate[0].assign_vector(tmp_inter) 
        if input1.value.total_size() > input2.value.total_size():
            @for_range_opt(len1//len2)
            def _(i):
                v3 = v1.get_vector(i*len2, len2) * tmp_inter
                output.value.assign_vector(v3, i*len2)
        else:
            @for_range_opt(len2//len1)
            def _(i):
                v3 = v1.get_vector(0, len1) * operation.intermediate[0].get_vector(i*len1, len1)
                output.value.assign_vector(v3, i*len1)  
        op_id += 1# record the input and output of the op
    return output


def ops_mul_constant(self, c):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]
        dl_dself[:] += c * dl_dx[:]
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        if isinstance(self.value, Array):
            new_value = Array(self.value.length, self.value.value_type)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        if self.req_grad:
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]

        output.value[:] = input.value[:] * c

        op_id += 1
        # record the input and output of the op
        return output


def mat_mul(self, other):
    # record the input and output of the op
    return 0


def ops_add_constant(self, c):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]
        dl_dself[:] += dl_dx[:]
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    if prepare:
        if isinstance(self.value, Array):
            new_value = Array(self.value.length, self.value.value_type)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        if self.req_grad:
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
        else:
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]

        output.value[:] = input.value[:] + c

        op_id += 1
        # record the input and output of the op
        return output

class Tensor():
    check_indices = True
    def __init__(self, value, value_type=None, name=None, req_grad=False, grad=None):
        assert isinstance(value, Array) or isinstance(value, MultiArray) or isinstance(value, list)
        assert isinstance(grad, Array) or isinstance(grad, MultiArray) or grad is None
        if isinstance(value, list):
            if len(value) == 0 or value_type is None:
                raise CompilerError("the shape of a tensor must be a not-null list and value type must be determined")
            if len(value) == 1:
                self.value = Array(value[0], value_type)
            if len(value) > 1:
                self.value = MultiArray(value, value_type)
            self.shape = tuple(value)
        else:
            self.value = value
            self.shape = value.sizes
        self.name = name or fresh_name()
        self.value_type = self.value.value_type
        self.req_grad = req_grad
        self.sub_cache = {}
        if grad is not None:
            self.grad = grad
            dl_d[name] = self.grad
        else:
            if is_train and req_grad:
                self.grad = self.value.same_shape()
                self.grad.assign_all(0)
                dl_d[self.name] = self.grad
            else:
                self.grad = None
        tensors[self.name] = self

    def numel(self):
        return self.value.length

    def set_req_grad(self, req_grad):
        self.req_grad = req_grad

    @property
    def sizes(self):
        return self.value.sizes

    @staticmethod
    def disable_index_checks():
        Tensor.check_indices = False

    @property
    def dim(self):
        return len(self.value.sizes)
    
    def print_reveal_nested(self):
        self.value.print_reveal_nested()

    def grad_print_reveal_nested(self):
        if not self.req_grad:
            raise CompilerError("the tensor " + self.name +" has no gradient")
        self.grad.print_reveal_nested()

    def __repr__(self):
        return self.value
    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.

    @staticmethod
    def constant(value, name=None):
        var = Tensor(value, name)
        return var

    def backward(self):
        global prepare
        if prepare:
            return 0
        if not self.req_grad:
            return 0
        length = len(gradient_operation)
        index = 0
        dl_d[self.name].assign_all(1)
        # the following loop only runs once in the training process due the semantice of @for_range

        def gather_grad(entries):
            return [dl_d[entry] for entry in entries]
        for i in range(0, length):
            if self.name in gradient_operation[length-i-1].outputs:
                index = length - i
        for i in range(0, index):
            entry = gradient_operation[index-i-1]
            dl_doutputs = gather_grad(entry.outputs)
            entry.propagate(dl_doutputs, entry)
        return 0

    # Multiplication of a Variable, tracking gradients
    def __mul__(self, other):
        # todo, dimension of self and other may not be the same
        if isinstance(other, (int, float)):
            return ops_mul_constant(self, other)
        return element_wise_mul(self, other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, index):
        """ Part access.

        :param index: public (regint/cint/int)
        :return: :py:class:`Array` if one-dimensional, :py:class:`SubMultiArray` otherwise"""
        if isinstance(index, slice) and index == slice(None):
            return self
        if isinstance(index, int) and index < 0:
            index += self.sizes[0]
        key = program.curr_block, str(index)
        if key not in self.sub_cache:
            if util.is_constant(index) and \
               (index >= self.sizes[0] or index < 0):
                raise CompilerError('index out of range')
            elif self.check_indices:
                library.runtime_error_if(index >= self.sizes[0],
                                         'overflow: %s/%s',
                                         index, self.sizes)
            if len(self.sizes) == 2:
                new_value = \
                    Array(self.sizes[1], self.value.value_type,
                          self.value.address + index * self.sizes[1] *
                          self.value.value_type.n_elements() *
                          self.value.value_type.mem_size())
                if self.req_grad:
                    new_grad = \
                        Array(self.sizes[1], self.grad.value_type,
                              self.grad.address + index * self.sizes[1] *
                              self.grad.value_type.n_elements() *
                              self.grad.value_type.mem_size())
                else:
                    new_grad = None
            else:
                new_value = \
                    MultiArray(self.sizes[1:], self.value.value_type,
                                  self.value.address, index, debug=self.debug)
                if self.req_grad:
                    new_grad = \
                        MultiArray(self.sizes[1:], self.grad.value_type,
                                      self.grad.address, index, debug=self.debug)
                else:
                    new_grad = None
        else:
            res = self.sub_cache[key]
            return res
        res = Tensor(new_value, req_grad=self.req_grad, grad=new_grad)
        self.sub_cache[key] = res
        res.check_indices = self.check_indices
        return res

    def __setitem__(self, index, other):
    #     """ Part assignment.

    #     :param index: public (regint/cint/int)
    #     :param other: container of matching size and type """

        self.value[index] = other         
        

    @staticmethod
    def ones(sizes: list, value_type = sfix):
        print(sizes)
        assert isinstance(sizes, list)
        if len(sizes) == 0 or value_type is None:
            raise CompilerError("the shape of a tensor must be a not-null list and value type must be determined")
        if len(sizes) == 1:
            res_value = Array(sizes[0], value_type)
        if len(sizes) > 1:
            res_value = MultiArray(sizes, value_type)
        res_value.assign_all(1)
        res = Tensor(res_value)        
        return res

    @staticmethod
    def zeros(sizes: list, value_type = sfix):
        assert isinstance(sizes, list)
        if len(sizes) == 0 or value_type is None:
            raise CompilerError("the shape of a tensor must be a not-null list and value type must be determined")
        if len(sizes) == 1:
            res_value = Array(sizes[0], value_type)
        if len(sizes) > 1:
            res_value = MultiArray(sizes, value_type)
        res_value.assign_all(0)
        res = Tensor(res_value)        
        return res

    @staticmethod
    def eye(m, n =None, value_type = sfix):
        assert n is None or m == n
        res_value = MultiArray([m, m], value_type)
        @for_range(m)
        def _(i):
            res_value[i][i] = 1
        res = Tensor(res_value)    
        return res

    def random_initialize(self):
        return 0
    
    def mv(self, other,out=None):
        # mul of Two-dimension * Array,return an output,whose type is Tensor and value is Array
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy,=dl_doutputs
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
           
            input2_Matrix=Matrix(input2.shape[0],1,input2.value.value_type,address=input2.value.address)
            save1_sizes=input1.shape
            input1.value.view(reduce(operator.mul,save1_sizes[:-1]),save1_sizes[-1])
            # compute dB=A^T*dC+dB
            # compute dA=dC*B^T+dA
            dl_d[operation.inputs[0]].view(reduce(operator.mul,save1_sizes[:-1]),save1_sizes[-1])
            if isinstance(dl_dy,Array):                
                dl_dy_Matrix=Matrix(dl_dy.sizes[0],1,dl_dy.value_type,address=dl_dy.address)
                
                input1.value.trans_mul_add_to(dl_dy_Matrix,dl_d[operation.inputs[1]])
                
                dl_dy_Matrix.mul_trans_add_to(input2_Matrix,dl_d[operation.inputs[0]])
            else:
                save2_sizes=dl_dy.sizes
                dl_dy.view(dl_dy.length,1)
                input1.value.trans_mul_add_to(dl_dy,dl_d[operation.inputs[1]])
                dl_dy.mul_trans_add_to(input2_Matrix,dl_d[operation.inputs[0]])
                dl_dy.view(*list(save2_sizes))
            dl_d[operation.inputs[0]].view(*list(save1_sizes))
            input1.value.view(*list(save1_sizes))
            
        global op_id
        if prepare:
            assert self.value.value_type==other.value.value_type,"Invalid Data Type"
            assert isinstance(self.value,MultiArray) and isinstance(other.value,Array),"The first parameter is Not MultiArray or the second parameter is not Array"
            assert len(other.shape)==1 and self.shape[-1]==other.shape[0],"Invalid Dimension"
            if out:
                new_value=out
            elif len(self.shape)==2:
                new_value = Array(self.shape[0], self.value.value_type)
            else:
                new_value = MultiArray(list(self.shape[:-1]), self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input1 = tensors[inputs[0]]
            input2 = tensors[inputs[1]]
            output = tensors[outputs[0]]
            input1.value.mv(input2.value,output.value)
            # n_threads=10 if input1.shape[0]>=1000 else 1
            # @multithread(n_threads,input1.shape[0])
            # def _(base, size):
            #     output.value.assign_part_vector(input1.value.direct_mul(input2.value,indices=(regint.inc(size,base=base),regint.inc(input1.shape[1]), regint.inc(input2.shape[0]),regint.inc(1))),base)
            op_id += 1# record the input and output of the op
        return output
    
    def mm(self, other): #Two-dimension * two-dimension,return an output,whose type is Tensor.
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            input1 = tensors[operation.inputs[0]]
            input2 = tensors[operation.inputs[1]]
            if self.req_grad:
                dl_dy.mul_trans_add_to(input2.value,dl_d[operation.inputs[0]],n_threads=10 if input1.shape[0]>=1000 else 1)
                # C=AB partial derivate of dA=dC*B^T+dA
            if other.req_grad:
                input1.value.trans_mul_add_to(dl_dy,dl_d[operation.inputs[1]],n_threads=10 if input1.shape[0]>=1000 else 1)
                # C=AB partial derivate of dB=A^T*dC+dB
        global op_id
        if prepare:
            assert self.value.value_type==other.value.value_type,"Invalid Data Type"
            assert len(self.shape)==len(other.shape)==2 and self.shape[1]==other.shape[0],"Invalid Dimension"
            new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input1 = tensors[inputs[0]]
            input2 = tensors[inputs[1]]
            output = tensors[outputs[0]]
            input1.value.mm(input2.value, output.value)
            op_id += 1  # record the input and output of the op
        return output

    def single_bmm(self, other: 'Tensor') -> 'Tensor':
        '''
        Performs a batch matrix-matrix product of matrices stored in self and other.
        Note: This function does not broadcast
        :param self.shape: [*b, n, m]
        :param other.shape: [m, p]
        :return: return.shape: [*b, n, p]
        '''
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            input1, input2 = tensors[operation.inputs[0]], tensors[operation.inputs[1]]
            if self.req_grad:
                cur_dinput1 = dl_dy.single_bmm_trans_to(input2.value)
                dl_d[operation.inputs[0]][:] += cur_dinput1[:]
                cur_dinput1.delete()
            if other.req_grad:
                # shenhao: need to revise permute
                cur_dinput2 = input1.value.trans_bmm_to(dl_dy, is_reduce=True)
                dl_d[operation.inputs[1]][:] += cur_dinput2[:]
                cur_dinput2.delete()
        # forward
        global op_id
        if prepare:
            assert len(self.sizes) >= 3 and self.sizes[-1] == other.sizes[0], "Invalid Dimension"
            batch, n, p = self.sizes[:-2], self.sizes[-2], other.sizes[-1]
            output = Tensor(MultiArray([*batch, n, p], other.value.value_type), req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs, outputs = operation.inputs, operation.outputs
            input1, input2, output = tensors[inputs[0]], tensors[inputs[1]], tensors[outputs[0]]
            input1.value.single_bmm(input2.value, output.value)
            op_id += 1
        return output
    
    def bmm(self, other: 'Tensor') -> 'Tensor':
        '''
        Performs a batch matrix-matrix product of matrices stored in self and other.
        Note: This function does not broadcast
        :param self.shape: [*b, n, m]
        :param other.shape: [*b, m, p]
        :return: return.shape: [*b, n, p]
        '''
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            input1, input2 = tensors[operation.inputs[0]], tensors[operation.inputs[1]]
            if self.req_grad:
                cur_dinput1 = dl_dy.bmm_trans_to(input2.value)
                dl_d[operation.inputs[0]][:] += cur_dinput1[:]
                cur_dinput1.delete()
            if other.req_grad:
                cur_dinput2 = input1.value.trans_bmm_to(dl_dy)
                dl_d[operation.inputs[1]][:] += cur_dinput2[:]
                cur_dinput2.delete()
        # forward
        global op_id
        if prepare:
            assert len(self.sizes) == len(other.sizes) >= 3 and self.sizes[:-2] == other.sizes[:-2] and self.sizes[-1] == other.sizes[-2], "Invalid Dimension"
            batch, n, p = self.sizes[:-2], self.sizes[-2], other.sizes[-1]
            output = Tensor(MultiArray([*batch, n, p], other.value.value_type), req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs, outputs = operation.inputs, operation.outputs
            input1, input2, output = tensors[inputs[0]], tensors[inputs[1]], tensors[outputs[0]]
            input1.value.bmm(input2.value, output.value)
            op_id += 1
        return output

    def dot(self, other):
        #Mul of two Array 
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy,=dl_doutputs
            dl_d[operation.inputs[0]][:]+= tensors[operation.inputs[1]].value[:]*dl_dy #dA=dC*B+dA
            dl_d[operation.inputs[1]][:]+= tensors[operation.inputs[0]].value[:]*dl_dy #dB=dC*A+dB
        global op_id
        if prepare:
            assert self.value.value_type==other.value.value_type,"Invalid Data Type"
            assert isinstance(self.value,Array) and isinstance(other.value,Array),"Not Array error"
            assert self.shape[0]==other.shape[0],"Invalid Dimension"
            new_value=Array(1,self.value.value_type)
            output=Tensor(new_value,req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation=Operation(inputs=[self.name,other.name],outputs=[output.name],propagate=propagate)
            else:
                operation=Operation(inputs=[self.name,other.name],outputs=[output.name],propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            output=tensors[operation.outputs[0]]
            @for_range(self.shape[0])
            def _(i):
                output.value[0]+=(input1.value[i]*input2.value[i])
            op_id+=1
        return output
    
    def matmul(self, other):
        assert self.dim >= other.dim, "The former must be higher dimensional than the latter"
        if self.dim == other.dim:
            if self.dim == 1:
                return self.dot(other)
            elif self.dim == 2:
                return self.mm(other)
            else:
                return self.bmm(other)
        else:
            if other.dim == 1:
                return self.mv(other)
            elif other.dim == 2:
                return self.single_bmm(other)
            else:
                raise CompilerError("Invalid Dimension: The multiplication does not match")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ops_add_constant(self, other)
        return element_wise_add(self, other)
    

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return ops_add_constant(self, -other)
        return element_wise_sub(self, other)

    def __neg__(self):
        return ops_mul_constant(self, -1)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ops_mul_constant(self, 1./other)
        return element_wise_div(self, other)

    def view(self, sizes):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare:
            product = reduce(lambda x, y: x*y, self.shape)
            if isinstance(sizes, int):
                assert sizes == product, "Invalid Dimension"
                new_value = Array(sizes, self.value.value_type)
            else:
                assert all(isinstance(x, int) and x > 0 for x in sizes), "Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1) == 1, "-1 Occurs More than Once "
                    tmp = reduce(lambda x, y: x*y, sizes)
                    assert product % (-tmp) == 0, "Invalid Dimension"
                    sizes[sizes.index(-1)] = product/(-tmp)
                new_value = MultiArray(sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            output.value.assign(self.value)
            op_id += 1
        return output

    def squeeze(self, dim=None):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare:
            if dim:
                new_sizes = list(self.shape)
                assert dim < len(self.shape), "Invalid Dimension"
                del new_sizes[dim]
            else:
                new_sizes = [x for x in self.shape if x != 1]
            if len(new_sizes) > 1:
                new_value = MultiArray(new_sizes, self.value.value_type)
            else:
                assert len(new_sizes) == 1 and new_sizes[0] > 0, "Invalid Dimension"
                new_value = Array(new_sizes[0], value_type=self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            output.value.assign(self.value)
            op_id += 1
        return output

    def unsqueeze(self, dim):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_d[operation.inputs[0]].assign(dl_doutputs[0])
        global op_id
        if prepare:
            new_sizes = list(self.shape)
            assert isinstance(dim, int) and dim < len(self.shape) and dim >= -len(self.shape), "Invalid Dimension"
            new_sizes.insert(dim, 1)
            new_value = MultiArray(new_sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            output.value.assign(self.value)
            op_id += 1
        return output

    def gather(self):
        # todo
        return self

    def reshape(self, sizes):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare:
            product = reduce(lambda x, y: x*y, self.shape)
            if isinstance(sizes, int):
                assert sizes == product, "Invalid Dimension"
                new_value = Array(sizes, self.value.value_type)
            else:
                assert all(isinstance(x, int) and x > 0 for x in sizes), "Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1) == 1, "-1 Occurs More than Once "
                    tmp = reduce(lambda x, y: x*y, sizes)
                    assert product % (-tmp) == 0, "Invalid Dimension"
                    sizes[sizes.index(-1)] = product/(-tmp)
                new_value = MultiArray(sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            output.value.assign(self.value)
            op_id += 1
        return output

    def permute(self, new_perm):  # todo :这里的参数不应该是list类型的new-perm，而应该是*newperm :pytorch中：x.permute(2, 0, 1)
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            L=len(self.shape)
            inv_new_perm=[None]*L
            for i in range(L):
                inv_new_perm[new_perm[i]]=i #s2[s1[i]]=i
            dl_dy.permute_without_malloc(dl_d[operation.inputs[0]],inv_new_perm)
        global op_id
        if prepare:
            assert isinstance(self.value, MultiArray), "Error,Permute operation must be MultiArray"  # 置换维度，那么肯定是MultiArray吧
            target_size = self.value.tuple_permute(self.shape, new_perm)  # just for calling of tuple_permute function
            new_value = MultiArray(target_size, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            self.value.permute_without_malloc(output.value, new_perm)  # output的值在参数中传入后被修改
            op_id += 1
        return output

    def transpose(self):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            if isinstance(dl_doutputs[0], Array):
                dl_d[operation.inputs[0]][:] += dl_doutputs[0][:]
            else:
                dl_d[operation.inputs[0]][:] += dl_doutputs[0].transpose()[:]
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.shape[0], self.value.value_type)
            else:
                assert len(self.value.sizes) == 2, 'Invalid dimension'
                new_sizes = [self.value.sizes[1], self.value.sizes[0]]
                new_value = MultiArray(new_sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            input = tensors[operation.inputs[0]]
            output = tensors[operation.outputs[0]]
            if len(self.shape) == 1:  # in this case:Array
                output.value[:] = input.value[:]
            else:
                output.value = input.value.transpose()
            op_id += 1
        return output

    def concate(self, other, axis=0):  # 按照axis指定维度进行拼接
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            size_pre=reduce(lambda x,y:x*y,input1.shape[axis:])
            size_next=reduce(lambda x,y:x*y,input2.shape[axis:]) 
            if input1.req_grad and input2.req_grad:
                @for_range(input1.value.length//size_pre)
                def _(i):    
                    input1.grad.assign_vector(dl_doutputs[0].get_vector(i*size_pre,size_pre),i*size_pre)
                    input2.grad.assign_vector(dl_doutputs[0].get_vector(i*size_next,size_next),i*size_next)
            elif input1.req_grad:
                @for_range(input1.value.length//size_pre)
                def _(i):
                    input1.grad.assign_vector(dl_doutputs[0].get_vector(i*size_pre,size_pre),i*size_pre)
            elif input2.req_grad:
                @for_range(input1.value.length//size_pre)
                def _(i):
                    input2.grad.assign_vector(dl_doutputs[0].get_vector(i*size_next,size_next),i*size_next)                
        global op_id
        if prepare:
            assert self.value.value_type is other.value.value_type, "Invalid value_type"
            if isinstance(self.value, Array) and isinstance(other.value, Array):
                target_len = self.value.length + other.value.length
                new_value = Array(target_len, self.value.value_type)
            else:
                assert len(self.shape)==len(other.shape),"Inequal Dimension"
                for i in range(len(self.shape)):
                    if i != axis and self.shape[i] != other.shape[i]:
                        raise ValueError("Invalid Dimension")
                target_size = other.value.shape
                target_size[axis] += self.value.shape[axis]
                new_value = MultiArray(target_size, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            size_pre=reduce(lambda x,y:x*y,self.shape[axis:])
            size_next=reduce(lambda x,y:x*y,other.shape[axis:])
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            output=tensors[operation.outputs[0]]
            index=regint(0)    
            @for_range(input1.value.length//size_pre)
            def _(i):  
                #can not convert this to @for_range for the error info of "local variable 'index' referenced before assignment"
                output.value.assign_vector(input1.value.get_vector(i*size_pre,size_pre),index)
                index.update(index+size_pre)
                output.value.assign_vector(input2.value.get_vector(i*size_next,size_next),index)
                index.update(index+size_next)
            op_id+=1
        return output

    def abs(self):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += (2 * inter[:] - 1) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            c = input.value[:] > 0
            operation.intermediate[0].assign_vector(c)  # write to mem

            output.value[:] = (2*c-1) * input.value[:]
            op_id += 1
        # record the input and output of the op
        return output

    def exp(self):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += inter[:] * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            ex = mpc_math.pow_fx(math.e, input.value[:])
            operation.intermediate[0].assign_vector(ex)

            output.value[:] = ex
            op_id += 1
        # record the input and output of the op
        return output

    def log(self, base=math.e):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += 1 / (self.value[:] * np.log(base)) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            output.value[:] = mpc_math.log_fx(input.value[:], base)

            op_id += 1
        # record the input and output of the op
        return output

    def pow(self, pow):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += pow * mpc_math.pow_fx(self.value[:], pow-1) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            output.value[:] = mpc_math.pow_fx(input.value[:], pow)

            op_id += 1
        # record the input and output of the op
        return output

    def cos(self):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):  # dl_outputs is Tensor.value
            dl_dx, = dl_doutputs
            dl_dself = dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*(-mpc_math.sin(self.value[:]))
        global op_id
        if prepare:
            if isinstance(self.value, Array):  # Array is instance of tensor?
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]  # input is Tensor
            output = tensors[outputs[0]]
            output.value[:] = mpc_math.cos(input.value[:])
            op_id += 1
            return output

    def sin(self):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):  # dl_outputs is Tensor.value
            dl_dx, = dl_doutputs
            dl_dself = dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*mpc_math.cos(self.value[:])
        global op_id
        if prepare:
            if isinstance(self.value, Array):  # Array is instance of tensor?
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]  # input is Tensor
            output = tensors[outputs[0]]
            output.value[:] = mpc_math.sin(input.value[:])
            op_id += 1
            return output

    def mean(self, dim=0):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += dl_dx[0] / self.value.total_size()
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_sizes = 1
            else:
                new_sizes = self.sizes[:dim] +  self.sizes[dim+1:]
            if len(new_sizes) <= 1:
                new_value = Array(new_sizes[0], self.value_type)
            else:
                new_value = MultiArray(new_sizes, self.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            # output.value[:] = sum(input.value[:]) / self.value.total_size()
            index_groups = input.value.getIndexGroups_by_dim(dim)
            for i in range(len(index_groups)):
                summary = input.value.value_type(0)
                for j in index_groups[i]:
                    summary += input.value.get_vector_by_indices(*j)
                output.value.assign_vector(summary, i)
            if isinstance(input.value, Array):
                output.value[:] /= cint(input.value.size)
            else:
                output.value[:] /= cint(input.value.sizes[dim])
            op_id += 1
        # record the input and output of the op
        return output

    def sum(self, dim=0):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += dl_dx[0]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                new_sizes = (1,)
            else:
                new_sizes = self.sizes[:dim] +  self.sizes[dim+1:]
            if len(new_sizes) <= 1:
                new_value = Array(new_sizes[0], self.value_type)
            else:
                new_value = MultiArray(new_sizes, self.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            # output.value[:] = sum(input.value[:]) 
            index_groups = input.value.getIndexGroups_by_dim(dim)
            for i in range(len(index_groups)):
                summary = input.value.value_type(0)
                for j in index_groups[i]:
                    summary += input.value.get_vector_by_indices(*j)
                output.value.assign_vector(summary, i)
            op_id += 1
        # record the input and output of the op
        return output

    def std(self):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dmean = operation.intermediate[0]
            stdvalue = operation.intermediate[1]
            dl_dself = dl_d[inputs[0]]

            dl_dself[:] += dl_dx[0] / stdvalue[0] / (self.value.total_size()-1) * dmean[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            new_value = Array(1, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)

            if isinstance(self.value, Array):
                inter1 = Array(self.value.length, self.value.value_type)
            else:
                inter1 = MultiArray(self.value.sizes, self.value.value_type)
            inter2 = Array(1, self.value.value_type)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter1, inter2])
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter1, inter2])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            mean = sum(input.value[:]) / self.value.total_size()
            dmean = input.value[:] - mean
            stdvalue = mpc_math.sqrt(sum(dmean ** 2) / (self.value.total_size()-1))

            operation.intermediate[0].assign_vector(dmean)
            operation.intermediate[1].assign_vector(stdvalue)
            output.value[:] = stdvalue

            op_id += 1
        # record the input and output of the op
        return output

    def var(self):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dmean = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]

            dl_dself[:] += 2 / (self.value.total_size()-1) * dmean[:] * dl_dx[0]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            new_value = Array(1, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if isinstance(self.value, Array):
                inter = Array(self.value.length, self.value.value_type)
            else:
                inter = MultiArray(self.value.sizes, self.value.value_type)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]

            mean = sum(input.value[:]) / self.value.total_size()
            dmean = input.value[:] - mean
            output.value[:] = sum(dmean ** 2) / (self.value.total_size()-1)

            operation.intermediate[0].assign_vector(dmean)

            op_id += 1
        # record the input and output of the op
        return output

    def norm(self, dim=None, keepdim=False):
        pass

    def softmax(self, dim=-1):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            output = tensors[operation.outputs[0]]
            if self.req_grad:
                if isinstance(self.value, MultiArray):
                    inter_mul1, inter_mul2, inter_sum = operation.intermediate[-3], operation.intermediate[-2], operation.intermediate[-1]
                    # dl_dx = softmax(x)*(dl_dy-(dl_dy*softmax(x)).sum(dim=-1))
                    dl_dy.element_wise_mul(output.value, inter_mul1)
                    inter = dl_dy - inter_mul1
                    inter.sum(dim, res=inter_sum, keepdims=True)
                    output.value.element_wise_mul(inter_sum, inter_mul2)
                    dl_d[operation.inputs[0]][:] += inter_mul2[:]
                    inter.delete()
                else:
                    res = output.value[:]*(dl_dy[:]-sum(output.value[:]*dl_dy[:]))
                    dl_d[operation.inputs[0]][:] += res
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):
                output = Tensor(Array(self.sizes[0], self.value_type), req_grad=self.req_grad)
                inter = []
            else:
                output = Tensor(MultiArray(self.sizes, self.value_type), req_grad=self.req_grad)
                new_sizes = [*self.sizes]
                new_sizes[dim], new_sizes[-1] = new_sizes[-1], new_sizes[dim]
                per_x, per_res = MultiArray(new_sizes, self.value_type), MultiArray(new_sizes, self.value_type)
                inter = [per_x, per_res]
            if self.req_grad:
                if isinstance(self.value, MultiArray):
                    new_sizes = self.sizes[:dim] + self.sizes[dim+1:]
                    new_sizes = new_sizes + (1,) if len(new_sizes) == 1 else new_sizes
                    inter_mul1, inter_mul2, inter_sum = MultiArray(self.value.sizes, self.value.value_type), MultiArray(
                        self.value.sizes, self.value.value_type), MultiArray(new_sizes, self.value.value_type)
                    inter += [inter_mul1, inter_mul2, inter_sum]
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=inter)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=inter)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            op_id += 1
        else:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if isinstance(self.value, Array):
                softmax_last_dim(input.value, dim, output.value)
            else:
                softmax_last_dim(input.value, dim, output.value, [operation.intermediate[0],operation.intermediate[1]])
            op_id += 1
        # record the input and output of the op
        return output

    def relu(self):
        pass
    
    def sigmoid(self):
        pass

    def tanh(self):
        pass
    
    def size(self):
        return self.value.sizes

    def zero_grad(self):
        self.grad.assign_all(0)


# reset operation
def reset_gloabal_store():
    gradient_operation.clear()
    for key, item in tensors.items():
        item.value.delete()
    tensors.clear()
    for key, item in dl_d.items():
        item.delete()
    dl_d.clear()
    op_id_store.clear()

# call this function after each iteration
def reset_op_id():
    global op_id
    op_id = 0


def get_opid():
    global op_id
    return op_id


def set_opid(new_id):
    global op_id
    op_id = new_id


def add_operation(operation):
    global gradient_operation
    gradient_operation.append(operation)


def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name


def train():
    global prepare
    prepare = False

def get_prepare():
    global prepare
    return prepare

def untrain():
    global prepare
    prepare = True


def same_shape(sizes1, sizes2):
    if len(sizes1) != len(sizes2):
        return False
    for i in range(0, len(sizes1)):
        if sizes1[i] != sizes2[i]:
            return False
    return True


def autograd_function(func):
    def wrapper(*args, **kw):
        func(*args, **kw)
        untrain()
        reset_op_id()
        reset_gloabal_store()
    copy_doc(wrapper, func)
    return wrapper


def softmax_last_dim(x, dim=-1, res=None, inter=None):
    assert res is not None, "res must be specified"
    if isinstance(x, Array):
        # res = Array(x.length, x.value_type) if res is None else res
        return res.assign_vector(vec_softmax(x.get_vector()))
    else:
        assert inter is not None, "inter must be specified"
        # res = MultiArray(x.sizes, x.value_type) if res is None else res
        per_x, per_res = inter[0], inter[1]
        x.swap_single_dim(dim, -1, per_x)
        res.swap_single_dim(dim, -1, per_res)
        batch = per_x.sizes[:-1]
        n, m = reduce(operator.mul, batch) if len(batch) >= 2 else batch[0], per_x.sizes[-1]
        per_x.view(-1, per_x.sizes[-1]), per_res.view(-1, per_res.sizes[-1])
        index = regint(0)
        @for_range_opt(n, n)
        def _(i):
            per_res.assign_vector(vec_softmax(per_x.get_vector(i*m, m)), index)
            index.update(index+m)
        per_x.view(*batch, m), per_res.view(*batch, m)
        per_res.swap_single_dim(dim, -1, res)
        return res

def vec_softmax(x):
    e_x = mpc_math.exp_fx(x - util.max(x))
    return e_x / sum(e_x)

def broadcast(*args: Tensor) -> List[Tensor]:
    """
    This function broadcasts the input arguments to match the shape of each other.
    """
    shapes = [arg.shape for arg in args]
    broadcast_shape = compute_broadcast_shape(*shapes)
    return (expand_to_shape(arg, broadcast_shape) for arg in args)


def compute_broadcast_shape(*shapes: Tuple[int]) -> Tuple[int]:
    reversed_shapes = [shape[::-1] for shape in shapes]
    broadcast_shape = []
    for dims in zip_longest(*reversed_shapes, fillvalue=1):
        greater_than_one_dims = [dim for dim in dims if dim > 1]
        if len(set(greater_than_one_dims)) > 1:
            raise ValueError("operands could not be broadcast together with shapes " + ' '.join(map(str, shapes)))
        broadcast_shape.append(max(dims))
    return tuple(broadcast_shape[::-1])


def squeeze_first_dim(inp: Any, len: int = 1) -> Union[Array, MultiArray]:
    assert isinstance(inp, (sfix, cfix, sint, cint, regint, Array, SubMultiArray, MultiArray)), "Input must be a scale(sfix,cfix,sint,cint,regint) or a array(Array,SubMultiArray,MultiArray)"
    if isinstance(inp, (sfix, cfix, sint, cint, regint)):
        res = Array(len, type(inp))
        res.assign_all(inp)
    else:
        shape = (inp.length,) if isinstance(inp, Array) else inp.sizes
        res = MultiArray([len, *shape], inp.value_type)
        for i in range(len):
            res[i] = inp
    return res


def expand_to_shape(inp: Tensor, target_shape: Tuple[int]) -> Tensor:
    """
    This function expands the inp to match the target_shape using broadcasting rules.
    """
    assert isinstance(inp, Tensor), "Input must be a Tensor"
    input_shape = inp.shape
    input = inp.value
    # Calculate the difference in dimensions between the input and target
    diff_dim = len(target_shape) - len(input_shape)

    # If the input tensor has fewer dimensions than target shape, add dimensions to the front
    if diff_dim > 0:
        for _ in range(diff_dim):
            input = squeeze_first_dim(input)

    res = MultiArray(list(target_shape), input.value_type)

    def expand_dim(obj: Union[Array, MultiArray], res: MultiArray, dim: int) -> Union[Array, MultiArray]:
        """
        This is a recursive helper function to expand the list along the specified dimension.
        """
        # If the current dimension is less than the number of dimensions in target shape
        if dim >= len(target_shape):
            return obj

        # Get the shape of the current input tensor
        current_shape = (obj.length,) if isinstance(obj, Array) else obj.sizes
        # If the size at the current dimension is 1, replicate the element to match target size
        if current_shape[0] == 1 and target_shape[dim] != 1:
            obj = squeeze_first_dim(obj[0], target_shape[dim])
        # Continue to expand each item in the current list if not in the last dimension
        if dim + 1 < len(target_shape):
            for i in range(target_shape[dim]):
                res[i] = expand_dim(obj[i], res[i], dim + 1)
            return res
        else:
            return obj

    return Tensor(expand_dim(input, res, 0))
