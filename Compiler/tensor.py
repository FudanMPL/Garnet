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
from Compiler import graph_visualization
# from Compiler.GC.types import sbitintis_train
from functools import reduce
from typing import List, NamedTuple, Callable, Dict, Optional, Union, Tuple, Any
from ml import approx_sigmoid, argmax
_name = 1


class Operation(NamedTuple):
    inputs: List[str]
    outputs: List[str]

    # apply chain rule
    propagate: 'Callable[List[Tensor], List[Tensor]]'
    # forward : 'Callable[List[Tensor], List[Tensor]]'
    intermediate: List = []
    name: str = 'op'


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
# init_op_id is the index of the first operation of model forward
init_op_id = 0
#forward indicates whether starting model forward
forward = False
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

def get_permute(n, dims):
    perm = list(filter(lambda x: x not in dims, range(n))) + dims
    return tuple(perm)

def get_permute_back(n, dims):
    perm = list(filter(lambda x: x not in dims, range(n))) + dims
    perm_back = [0 for i in range(len(perm))]
    for i in range(len(perm)):
        perm_back[perm[i]] = i
    return tuple(perm_back)

def get_permute_d2front(n, dims):
    perm = dims + list(filter(lambda x: x not in dims, range(n)))
    return tuple(perm)

def check_subseq(li_self, li_other):
    # x = np.array(li_self)
    # y = np.array(li_other)
    
    # check_res = np.isin(x[x!=1], y[y!=1]).all()
    
    # if check_res:
    #     mask = np.isin(y, x)
    # else:
    #     mask = np.isin(x, y)
        
    # indices = list(np.where(mask)[0])
    # return check_res, indices
    a = li_self
    b = li_other
    
    if len(b) > len(a):
        return False, []
    
    for i in range(1, len(b) + 1):
        if b[-i] != 1 and b[-i] != a[-i]:
            return False, []

    positions = []

    for i in range(len(a) - len(b), len(a)):
        if b[-(len(a) - i)] == 1:
            # positions.append(i)
            pass
        elif b[-(len(a) - i)] == a[i]:
            positions.append(i)

    return True, positions

def reconst_dims(v1, v2):
    # v1, v2= input1.value, input2.value
    flag1, dim1 = check_subseq(v1.sizes, v2.sizes)
    flag2, dim2 = check_subseq(v2.sizes, v1.sizes)
    assert flag1 or flag2, "Invalid Dimension"
    # swap to ensure v1 size is bigger than v2 size
    dims = dim1
    if flag2 and not flag1:
        v1, v2 = v2, v1
        dims = dim2
    return dims, v1, v2

def aux_mul(input, other):
    if other < 10**-4:
        while other < 10**-4:
            input *= 10**-4
            other*=10**4
    else:
      input *= other
    return input
@buildingblock("add-forward")
def element_wise_add(self, other):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-add-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        temp1, temp2 = operation.intermediate
        dl_dself, dl_dother = (None, None)
        if self.req_grad:
            dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        if other.req_grad:
            dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        input1=tensors[operation.inputs[0]].value
        input2=tensors[operation.inputs[1]].value
        if input1.total_size()<input2.total_size():
            v1, v2 = v2, v1
            req_grad1, req_grad2 = req_grad2, req_grad1
            input1, input2 = input2, input1
        # v1 back directly 
        if req_grad1:
            v1[:] += dl_dx[:]
        # broadcasted v2 back with reduce
        if req_grad2:
            dims, input1, input2 = reconst_dims(input1, input2)
            dl_dx.permute_without_malloc(temp2, get_permute_d2front(len(input1.sizes), dims))
            dl_dx_pmt = temp2
            
            stride = input1.total_size()//input2.total_size()
            vsum = Array(1, dl_dx_pmt.value_type)
            
            @for_range(input2.total_size())
            def _(i):
                vsum.assign_all(0)
                @for_range(stride)
                def _(j):
                    vsum[:] += dl_dx_pmt.get_vector(i*stride+j, 1)
                # vsum = sum(dl_dx_pmt.get_vector(i*stride, stride))
                v2.assign_vector(v2.get_vector(i, 1)+vsum[:], i) 
            vsum.delete()
            break_point()
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, predict_value_type(self.value, other.value))
            else:
                new_value = MultiArray(other.value.sizes, predict_value_type(self.value, other.value))
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], predict_value_type(self.value, other.value))
            else:
                new_value = Array(other.value.sizes[0], predict_value_type(self.value, other.value))
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        dim, v1, v2 = reconst_dims(self.value, other.value)
        target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dim))
        temp1 = MultiArray(target_size, v1.value_type)
        target_size = v1.tuple_permute(v1.sizes, get_permute_d2front(len(v1.sizes), dim))
        temp2 = MultiArray(target_size, predict_value_type(self.value, other.value))
        # check whether require grad
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate,
                              intermediate=[temp1, temp2], name='add')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id

    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        temp1, temp2 = operation.intermediate
        if not forward:
            init_op_id += 1
        # # permute input for boardcasted
        # dims, v1, v2 = reconst_dims(input1.value, input2.value)
        # v1.permute_without_malloc(temp1, get_permute(len(v1.sizes), dims))
        # v1 = temp1
        
        # # element_wise_add
        # len1, len2 = v1.total_size(), v2.total_size()
        # @for_range(len1//len2)
        # def _(i):
        #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        #     temp2.assign_vector(v3, i*len2)
        # break_point()
        
        # # permute back
        # temp2.permute_without_malloc(output.value, get_permute_back(len(v1.sizes), dims))

        boardcasted_multiarray_add(input1.value, input2.value, output.value)
        
    op_id += 1# record the input and output of the op
    return output

@buildingblock("sub-forward")
def element_wise_sub(self, other):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-sub-backward")
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
    global init_op_id
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
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[temp_matrix], name='sub')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
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

def boardcasted_multiarray_add(v1, v2, output):
    # predict size and type for result
    dims, v1, v2 = reconst_dims(v1, v2)
    target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dims))  
    temp1 = MultiArray(target_size, v1.value_type)
    temp2 = MultiArray(target_size, predict_value_type(v1, v2))
    # permute input for boardcasted
    v1.permute_without_malloc(temp1, get_permute(len(v1.sizes), dims))
    v1 = temp1
    # vectorize add
    len1, len2 = v1.total_size(), v2.total_size()
    assert len1 % len2==0, "Invalid Dimension"
    @for_range_opt(len1//len2)
    def _(i):
        v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        temp2.assign_vector(v3, i*len2)
    break_point()
    # permute back
    temp2.permute_without_malloc(output, get_permute_back(len(v1.sizes), dims))
    # release mem
    temp1.delete()
    temp2.delete()

def boardcasted_multiarray_mul(v1, v2, output):
    # predict size and type for result
    dims, v1, v2 = reconst_dims(v1, v2)
    target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dims))  
    temp1 = MultiArray(target_size, v1.value_type)
    temp2 = MultiArray(target_size, predict_value_type(v1, v2))
    # permute input for boardcasted
    v1.permute_without_malloc(temp1, get_permute(len(v1.sizes), dims))
    v1 = temp1
    # vectorize mul
    len1, len2 = v1.total_size(), v2.total_size()
    assert len1 % len2==0, "Invalid Dimension"
    @for_range_opt(len1//len2)
    def _(i):
        v3 = v1.get_vector(i*len2, len2) * v2.get_vector(0, len2)
        temp2.assign_vector(v3, i*len2)
    break_point()
    # permute back
    temp2.permute_without_malloc(output, get_permute_back(len(v1.sizes), dims))
    # release mem
    temp1.delete()
    temp2.delete()
    
def boardcasted_multiarray_sub(v1, v2, output):
    # predict size and type for result
    dims, v1, v2 = reconst_dims(v1, v2)
    target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dims))  
    temp1 = MultiArray(target_size, v1.value_type)
    temp2 = MultiArray(target_size, predict_value_type(v1, v2))
    # permute input for boardcasted
    v1.permute_without_malloc(temp1, get_permute(len(v1.sizes), dims))
    v1 = temp1
    # vectorize sub
    len1, len2 = v1.total_size(), v2.total_size()
    assert len1 % len2==0, "Invalid Dimension"
    @for_range_opt(len1//len2)
    def _(i):
        v3 = v1.get_vector(i*len2, len2) - v2.get_vector(0, len2)
        temp2.assign_vector(v3, i*len2)
    break_point()
    # permute back
    temp2.permute_without_malloc(output, get_permute_back(len(v1.sizes), dims))
    # release mem
    temp1.delete()
    temp2.delete()

def predict_value_type(self, other):
    stype = self.value_type
    otype = other.value_type
    sflag = 0
    fflag = 0
    output_store = {
        (0, 0): cint,
        (0, 1): cfix,
        (1, 0): sint,
        (1, 1): sfix
    }
    if stype == sfix or otype == sfix or stype == cfix or otype == cfix:
        fflag = 1
    if stype == sint or otype == sint or stype == sfix or otype == sfix:
        sflag = 1
    return output_store[(sflag, fflag)]

def mixed_dot_product(v1, v2):
    assert len(v1) == len(v2)
    temp1 = MultiArray([1, len(v1)], type(v1))
    temp2 = MultiArray([len(v2), 1], type(v2))
    temp1.assign_vector(v1)
    temp2.assign_vector(v2)
    res = temp1 * temp2
    res_value = res[:]
    temp1.delete()
    temp2.delete()
    res.delete()
    return res_value
    
@buildingblock("mul-forward")
def element_wise_mul(self, other):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-mul-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        temp3, temp4, temp5 = operation.intermediate
        dl_dself, dl_dother = (None, None)
        if self.req_grad:
            dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        if other.req_grad:
            dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # swap to ensure v1 size is bigger than v2 size  
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        input1=tensors[operation.inputs[0]].value
        input2=tensors[operation.inputs[1]].value
        
        if input1.total_size()<input2.total_size():
            v1, v2 = v2, v1
            req_grad1, req_grad2 = req_grad2, req_grad1
            input1, input2 = input2, input1
            
        dims, input1, input2 = reconst_dims(input1, input2)
        # v1 back directly 
        if req_grad1:
            # dl_dx.permute_without_malloc(temp1, get_permute(len(dl_dx.sizes), dims))
            # dl_dx_pmt = temp1
            # stride = input2.total_size()
            # # temp3 = permute(dl_dx) * permute(input2.value)
            # @for_range_opt(input1.total_size()//input2.total_size())
            # def _(i):
            #     v3 = dl_dx_pmt.get_vector(i*stride, stride) * input2.get_vector(0, stride)
            #     temp2.assign_vector(v3, i*stride)
            # break_point()   
            # # v1 = permute_back(temp3)
            # temp2.permute_without_malloc(temp5, get_permute_back(len(input1.sizes), dims))
            # v1[:] += temp5[:]
            boardcasted_multiarray_mul(dl_dx, input2, temp5)
            v1[:] += temp5[:]
        # broadcasted v2 back with reduce
        if req_grad2:
            dl_dx.permute_without_malloc(temp3, get_permute_d2front(len(dl_dx.sizes), dims))
            input1.permute_without_malloc(temp4, get_permute_d2front(len(input1.sizes), dims))
            dl_dx_pmt, input1_pmt = temp3, temp4
            stride = input1.total_size()//input2.total_size()
            @for_range_opt(input2.total_size())
            def _(i):
                v3 = mixed_dot_product(dl_dx_pmt.get_vector(i*stride, stride), input1_pmt.get_vector(i*stride, stride))
                v2.assign_vector(v2.get_vector(i, 1)+v3, i)    
            break_point()
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        # check shape
        # assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
        if isinstance(self.value, MultiArray) or isinstance(other.value, MultiArray):
            if self.value.total_size()>other.value.total_size():
                new_value = MultiArray(self.value.sizes, predict_value_type(self.value, other.value))
            else:
                new_value = MultiArray(other.value.sizes, predict_value_type(self.value, other.value))
        else:
            if self.value.total_size()>other.value.total_size():
                new_value = Array(self.value.sizes[0], predict_value_type(self.value, other.value))
            else:
                new_value = Array(other.value.sizes[0], predict_value_type(self.value, other.value))
        output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
        
        dims, v1, v2 = reconst_dims(self.value, other.value)
        # target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dims))
        # temp1 = MultiArray(target_size, v1.value_type)
        # temp2 = MultiArray(target_size, predict_value_type(self.value, other.value))
        target_size = v1.tuple_permute(v1.sizes, get_permute_d2front(len(v1.sizes), dims))
        temp3 = MultiArray(target_size, predict_value_type(self.value, other.value))
        temp4 = MultiArray(target_size, v1.value_type)
        temp5 = MultiArray(v1.sizes, v1.value_type)
        # check whether require grad
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[temp3, temp4, temp5], name='mul')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1        
        # # permute input for boardcasted
        # dims, v1, v2 = reconst_dims(input1.value, input2.value)
        # v1.permute_without_malloc(inter, get_permute(len(v1.sizes), dims))
        # v1 = inter

        # len1, len2 = v1.total_size(), v2.total_size()
        # assert len1 % len2==0, "Invalid Dimension"
        # # for i in range(0, len1//len2):
        # #     v3 = v1.get_vector(i*len2, len2) + v2.get_vector(0, len2)
        # #     output.value.assign_vector(v3, i*len2)
        # @for_range_opt(len1//len2)
        # def _(i):
        #     v3 = v1.get_vector(i*len2, len2) * v2.get_vector(0, len2)
        #     v1.assign_vector(v3, i*len2)
        # break_point()
        
        # # permute back
        # v1.permute_without_malloc(output.value, get_permute_back(len(v1.sizes), dims))

        boardcasted_multiarray_mul(input1.value, input2.value, output.value)
        
    op_id += 1# record the input and output of the op
    return output

@buildingblock("div-forward")
def element_wise_div(self, other):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-div-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        output_value = tensors[operation.outputs[0]].value
        temp1, temp2, temp3, temp4, temp5, temp6, temp7 = operation.intermediate
        dl_dself, dl_dother = (None, None)
        if self.req_grad:
            dl_dself = dl_d[inputs[0]]  # partial derivate of r = 1
        if other.req_grad:
            dl_dother = dl_d[inputs[1]]  # partial derivate of r = 1
        
        # convert "input" as convert div to mul 
        v1, v2 = dl_dself, dl_dother
        req_grad1, req_grad2 = self.req_grad, other.req_grad
        # temp5.assign_vector(1 / tensors[operation.inputs[1]].value[:])
        # temp6.assign_vector(tensors[operation.inputs[0]].value[:] / tensors[operation.inputs[1]].value[:] / tensors[operation.inputs[1]].value[:])
        boardcasted_multiarray_mul(output_value, temp5, temp6)
        temp6.assign_vector(-1 * temp6[:])
        input2, input1 = temp5, temp6
        
        # swap to ensure v1 size is bigger than v2 size  
        if input1.total_size()<input2.total_size():
            v1, v2 = v2, v1
            req_grad1, req_grad2 = req_grad2, req_grad1
            input1, input2 = input2, input1
            
        dims, input1, input2 = reconst_dims(input1, input2)
        # v1 back directly 
        if req_grad1:
            dl_dx.permute_without_malloc(temp1, get_permute(len(dl_dx.sizes), dims))
            dl_dx_pmt = temp1
            stride = input2.total_size()
            # temp3 = permute(dl_dx) * permute(input2.value)
            @for_range_opt(input1.total_size()//input2.total_size())
            def _(i):
                v3 = dl_dx_pmt.get_vector(i*stride, stride) * input2.get_vector(0, stride)
                temp2.assign_vector(v3, i*stride)
            break_point()   
            # v1 = permute_back(temp3)
            temp2.permute_without_malloc(temp7, get_permute_back(len(v1.sizes), dims))
            v1[:] += temp7[:]
        # broadcasted v2 back with reduce
        if req_grad2:
            dl_dx.permute_without_malloc(temp3, get_permute_d2front(len(dl_dx.sizes), dims))
            input1.permute_without_malloc(temp4, get_permute_d2front(len(input1.sizes), dims))
            dl_dx_pmt, input1_pmt = temp3, temp4
            stride = input1.total_size()//input2.total_size()
            @for_range_opt(input2.total_size())
            def _(i):
                v3 = mixed_dot_product(dl_dx_pmt.get_vector(i*stride, stride), input1_pmt.get_vector(i*stride, stride))
                v2.assign_vector(v2.get_vector(i,1)+v3, i)    
            break_point()
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        # check shape
        # assert check_boardcast_size(self.value.sizes, other.value.sizes), "Invalid Dimension"
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
        
        dims, v1, v2 = reconst_dims(self.value, other.value)
        target_size = v1.tuple_permute(v1.sizes, get_permute(len(v1.sizes), dims))
        temp1 = MultiArray(target_size, v1.value_type)
        temp2 = MultiArray(target_size, v1.value_type)
        target_size = v1.tuple_permute(v1.sizes, get_permute_d2front(len(v1.sizes), dims))
        temp3 = MultiArray(target_size, v1.value_type)
        temp4 = MultiArray(target_size, v1.value_type)
        temp5 = MultiArray(v2.sizes, v2.value_type)
        temp6 = MultiArray(v1.sizes, v1.value_type)
        temp7 = MultiArray(v1.sizes, v1.value_type)
        # check whether require grad
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[temp1, temp2, temp3, temp4, temp5, temp6, temp7],
                              name='div')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        temp5 = operation.intermediate[4]
        if not forward:
            init_op_id += 1        
        # get inverse of input2
        temp5.assign_vector(1/input2.value[:])
        
        boardcasted_multiarray_mul(input1.value, temp5, output.value)
        
    op_id += 1# record the input and output of the op
    return output

@buildingblock("mulconstant-forward")
def ops_mul_constant(self, c):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-20]+"-mulconstant-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]
        dl_dself[:] += dl_dx[:] * c
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if isinstance(self.value, Array):
            new_value = Array(self.value.length, self.value.value_type)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              name='mul')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
        output.value[:] = input.value[:] * c

    op_id += 1
        # record the input and output of the op
    return output


def mat_mul(self, other):
    # record the input and output of the op
    return 0

@buildingblock("addconstant-forward")
def ops_add_constant(self, c):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-20]+"-addconstant-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]
        dl_dself[:] += dl_dx[:]
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if isinstance(self.value, Array):
            new_value = Array(self.value.length, self.value.value_type)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              name='addc')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
        output.value[:] = input.value[:] + c

    op_id += 1
        # record the input and output of the op
    return output

@buildingblock("sum-forward")
def sum_of_array(self):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-sum-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]
        dl_dself[:] += dl_dx[0]
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:    
        new_value = Array(1, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        operation = Operation(inputs=[self.name], outputs=[output.name], 
                              propagate=propagate, name='sum')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1        
        # output.value[:] = sum(input.value[:])
        output.value[:] = 0
        @for_range(input.value.total_size())
        def _(i):
            output.value[:] += input.value[i]
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("mean-forward")
def mean_of_array(self):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-mean-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself = dl_d[inputs[0]]

        dl_dself[:] += dl_dx[0] / self.value.total_size()
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:    
        new_value = Array(1, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate, name='mean')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
        output.value[:] = sum(input.value[:]) / self.value.total_size()
        
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("var-forward")
def var_of_array(self, unbiased=False):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-var-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dmean = operation.intermediate[0] # reuse the intervalue in mem
        dl_dself = dl_d[inputs[0]]
        
        factor = 2 * dmean[:] * dl_dx[0]
        if unbiased:
            factor *= 1 / self.value.total_size()
        else:
            factor *= 1 / (self.value.total_size()-1)
        
        dl_dself[:] += factor
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:    
        new_value = Array(1, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
            
        if isinstance(self.value, Array):
            inter = Array(self.value.length, self.value.value_type)
        else:
            inter = MultiArray(self.value.sizes, self.value.value_type)
        
        operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
        mean = sum(input.value[:]) / self.value.total_size()
        dmean = input.value[:] - mean
        output.value[:] = sum(dmean ** 2) 
        
        if unbiased:
            output.value[:] *= 1 / self.value.total_size()
        else:
            output.value[:] *= 1 / (self.value.total_size()-1)
        
        operation.intermediate[0].assign_vector(dmean)
        
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("std-forward")
def std_of_array(self):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-std-backward")
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
    global init_op_id
    if prepare:    
        new_value = Array(1, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        if isinstance(self.value, Array):
            inter1 = Array(self.value.length, self.value.value_type)
        else:
            inter1 = MultiArray(self.value.sizes, self.value.value_type)
        inter2 = Array(1, self.value.value_type)
        
        operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter1, inter2])
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        
        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input = tensors[inputs[0]]
        output = tensors[outputs[0]]
        if not forward:
            init_op_id += 1
        mean = sum(input.value[:]) / self.value.total_size()
        dmean = input.value[:] - mean
        stdvalue = mpc_math.sqrt(sum(dmean ** 2) / (self.value.total_size()-1))
        
        operation.intermediate[0].assign_vector(dmean)
        operation.intermediate[1].assign_vector(stdvalue)
        output.value[:] = stdvalue
        
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("sum-forward")
def sum_of_multiarray(self, dim, keepdim=False):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-sum-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        dl_dself = dl_d[operation.inputs[0]]
        input_perm, = operation.intermediate
        
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        @for_range(dl_dx.total_size())
        def _(i):
            @for_range(stride)
            def _(j):
                input_perm.assign_vector(dl_dx.get_vector(i, 1), i*stride+j)
        # permute back
        new_perm = get_permute_back(len(self.value.sizes), dim)
        input_perm.permute_without_malloc(dl_dself, new_perm)
        
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if not keepdim:
            new_sizes = [self.value.sizes[i] for i in list(filter(lambda x: x not in dim, range(len(self.value.sizes))))]
        else:
            new_sizes = [(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))]
        if len(new_sizes) <= 1:
            new_value = Array(new_sizes[0], self.value.value_type)
        else:
            new_value = MultiArray(new_sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        new_perm = get_permute(len(self.value.sizes), dim)
        target_size = self.value.tuple_permute(self.shape, new_perm)
        input_perm = MultiArray(target_size, self.value.value_type)
        
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate,
                              intermediate=[input_perm], name='sum')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        input_perm = operation.intermediate[0]

        new_perm = get_permute(len(input.value.sizes), dim)
        input.value.permute_without_malloc(input_perm, new_perm)
        if not forward:
            init_op_id += 1        
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        summary = Array(1, input_perm.value_type)
        
        @for_range_opt(input.value.total_size()//stride)
        def _(i):
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += input_perm.get_vector(i*stride+j, 1)
            # summary = sum(input_perm.get_vector(i*stride, stride))
            output.value.assign_vector(summary[:], i)
        summary.delete()
        break_point()
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("mean-forward")
def mean_of_multiarray(self, dim, keepdim=False):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-mean-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        dl_dself = dl_d[operation.inputs[0]]
        input_perm, temp = operation.intermediate

        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        @for_range(dl_dx.total_size())
        def _(i):
            @for_range(stride)
            def _(j):
                input_perm.assign_vector(dl_dx.get_vector(i, 1), i*stride+j)
        break_point()
        # input_perm[:] *= 1/ stride
        n_per_item = reduce(operator.mul, input_perm.sizes[1:])
        @multithread(1, input_perm.sizes[0], max(1, 100000 // n_per_item))
        def _(base, size):
            input_perm.assign_part_vector( aux_mul(input_perm.get_part_vector(base, size), (1/ stride)), base)
           
        # permute back
        new_perm = get_permute_back(len(self.value.sizes), dim)
        input_perm.permute_without_malloc(temp, new_perm)
        n_per_item = reduce(operator.mul, dl_dself.sizes[1:])
        @multithread(1, dl_dself.sizes[0], max(1, 100000 // n_per_item))
        def _(base, size):
            dl_dself.assign_part_vector( dl_dself.get_part_vector(base, size)+temp.get_part_vector(base, size), base)
                    
        # dl_dself[:] += temp[:]
        
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if not keepdim:
            new_sizes = [self.value.sizes[i] for i in list(filter(lambda x: x not in dim, range(len(self.value.sizes))))]
        else:
            new_sizes = [(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))]
        if len(new_sizes) <= 1:
            new_value = Array(new_sizes[0], self.value.value_type)
        else:
            new_value = MultiArray(new_sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        new_perm = get_permute(len(self.value.sizes), dim)
        target_size = self.value.tuple_permute(self.shape, new_perm)
        input_perm = MultiArray(target_size, self.value.value_type)
        
        temp1 = MultiArray(self.value.sizes, self.value.value_type)
        
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[input_perm, temp1], name='mean')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        input_perm = operation.intermediate[0]

        new_perm = get_permute(len(input.value.sizes), dim)
        input.value.permute_without_malloc(input_perm, new_perm)
        if not forward:
            init_op_id += 1        
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        summary = Array(1, input_perm.value_type)
        @for_range(input.value.total_size()//stride)
        def _(i):
            # summary = sum(input_perm.get_vector(i*stride, stride))
            # output.value.assign_vector(summary, i)
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += input_perm.get_vector(i*stride+j, 1)
            output.value.assign_vector(summary[:], i)
        summary.delete()
        tmp = 1 / stride
        @multithread(1, output.value.total_size())
        def _(base, size):
            output.value.assign_vector(aux_mul(output.value.get_vector(base, size), tmp), base)

        # output.value.reshape([(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))])
    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("var-forward")
def var_of_multiarray(self, dim, keepdim=False, unbiased=False):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-var-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        dl_dself = dl_d[operation.inputs[0]]
        input_perm, mean, dmean, dmean_sqr = operation.intermediate
        # dl_dself[:] += 2 / (self.value.total_size()-1) * dmean[:] * dl_dx[0]
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        @for_range(dl_dx.total_size())
        def _(i):
            @for_range(stride)
            def _(j):
                input_perm.assign_vector(dl_dx.get_vector(i, 1), i*stride+j)
        break_point()
        # input_perm[:] *= 2
        if unbiased:
            n_per_item = reduce(operator.mul, input_perm.sizes[1:])
            @multithread(1, input_perm.sizes[0], max(1, 100000 // n_per_item))
            def _(base, size):
                input_perm.assign_part_vector( aux_mul(input_perm.get_part_vector(base, size), (2/ stride)), base)
        else:
            n_per_item = reduce(operator.mul, input_perm.sizes[1:])
            @multithread(1, input_perm.sizes[0], max(1, 100000 // n_per_item))
            def _(base, size):
                input_perm.assign_part_vector( aux_mul(input_perm.get_part_vector(base, size), (2/ (stride-1))), base)
        n_per_item = reduce(operator.mul, input_perm.sizes[1:])
        @multithread(1, input_perm.sizes[0], max(1, 100000 // n_per_item))
        def _(base, size):
            input_perm.assign_part_vector( input_perm.get_part_vector(base, size)*dmean.get_part_vector(base, size), base)
        # input_perm[:] *= dmean[:]
        # permute back
        new_perm = get_permute_back(len(self.value.sizes), dim)
        tmp = dl_dself.same_shape()
        input_perm.permute_without_malloc(tmp, new_perm)
        n_per_item = reduce(operator.mul, dl_dself.sizes[1:])
        @multithread(1, dl_dself.sizes[0], max(1, 100000 // n_per_item))
        def _(base, size):
            dl_dself.assign_part_vector( dl_dself.get_part_vector(base, size)+tmp.get_part_vector(base, size), base)
        tmp.delete()   
        # dl_dself[:] += dmean[:]
        
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if not keepdim:
            new_sizes = [self.value.sizes[i] for i in list(filter(lambda x: x not in dim, range(len(self.value.sizes))))]
        else:
            new_sizes = [(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))]
        if len(new_sizes) <= 1:
            new_value = Array(new_sizes[0], self.value.value_type)
            mean = Array(new_sizes[0], self.value.value_type)
        else:
            new_value = MultiArray(new_sizes, self.value.value_type)
            mean = MultiArray(new_sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        # tmp = MultiArray(self.value.sizes, self.value.value_type)
        # dmean_sqr = MultiArray(self.value.sizes, self.value.value_type)
        
        new_perm = get_permute(len(self.value.sizes), dim)
        target_size = self.value.tuple_permute(self.shape, new_perm)
        input_perm = MultiArray(target_size, self.value.value_type)
        dmean = MultiArray(target_size, self.value.value_type)
        dmean_sqr = MultiArray(target_size, self.value.value_type)        
        operation = Operation(inputs=[self.name], outputs=[output.name], 
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[input_perm, mean, dmean, dmean_sqr],
                              name='var')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        input_perm, mean, dmean, dmean_sqr = operation.intermediate
        # pre-perm
        if not forward:
            init_op_id += 1        
        new_perm = get_permute(len(input.value.sizes), dim)
        input.value.permute_without_malloc(input_perm, new_perm)
        # mean
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        summary = Array(1, input_perm.value_type)
        @for_range_opt(output.value.total_size())
        def _(i):
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += input_perm.get_vector(i*stride+j, 1)
            mean.assign_vector(summary[:], i)
        summary.delete()
        break_point()
        ss = (1/ stride)
        mean[:] = aux_mul(mean[:], ss)
        # n_per_item = reduce(operator.mul, mean.sizes[1:])
        # @multithread(1, input_perm.sizes[0], max(1, 100000 // n_per_item))
        # def _(base, size):
        #     print(mean)
        #     mean.assign_part_vector( mean.get_part_vector(base, size)* (1/ stride), base)
        # dmean
        dmean_value = Array(stride, input_perm.value_type)
        @for_range_opt(output.value.total_size())
        def _(i):
            dmean_value.assign_all(0)
            @for_range(stride)
            def _(j):
                dmean_value[j] = input_perm.get_vector(i*stride+j, 1) - mean.get_vector(i, 1) 
            dmean.assign_vector(dmean_value[:], i*stride)
        dmean_value.delete()
        break_point()
        # var        
        dmean_sqr[:] = dmean[:] ** 2
        summary = Array(1, output.value_type)
        @for_range_opt(output.value.total_size())
        def _(i):
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += dmean_sqr.get_vector(i*stride+j, 1)
            output.value.assign_vector(summary[:], i)
        summary.delete()
        # output.print_reveal_nested()
        break_point()
        if unbiased:
            tmp = 1 / stride
            @multithread(1, output.value.total_size())
            def _(base, size):
                output.value.assign_vector(aux_mul(output.value.get_vector(base, size), tmp) , base)

        else:
            tmp = 1 / (stride-1)
            @multithread(1, output.value.total_size())
            def _(base, size):
                output.value.assign_vector(aux_mul(output.value.get_vector(base, size), tmp), base)

    op_id += 1
    # record the input and output of the op
    return output

@buildingblock("std-forward")
def std_of_multiarray(self, dim, keepdim=False):
    # backward
    @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-std-backward")
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        dl_dself = dl_d[operation.inputs[0]]
        input_perm, factor, dmean, dmean_sqr, std = operation.intermediate
        
        # dl_dself[:] += dl_dx[0] / stdvalue[0] / (self.value.total_size()-1) * dmean[:]
        # new_perm = get_permute_d2front(len(self.value.sizes), dim)
        # target_size = self.value.tuple_permute(self.shape, new_perm)
        # input_perm = MultiArray(target_size, self.value.value_type) 

        # print(dmean_sqr.sizes)
        # boardcasted_multiarray_mul(factor, dmean, input_perm, dmean_sqr)
        # dmean_sqr[:] /= stride - 1
        # dl_dself[:] += dmean_sqr[:]
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        factor[:] = dl_dx[:] / std[:]
        @for_range_opt(dl_dx.total_size())
        def _(i):
            @for_range_opt(stride)
            def _(j):
                input_perm.assign_vector(factor.get_vector(i, 1), i*stride+j)
        break_point()
        input_perm[:] *= 1/(stride - 1)
        input_perm[:] *= dmean[:]
        # permute back
        new_perm = get_permute_back(len(self.value.sizes), dim)
        input_perm.permute_without_malloc(dmean, new_perm)
        dl_dself[:] += dmean[:]
        
        dl_dinputs = [dl_dself]
        return dl_dinputs
    # forward
    global op_id
    global init_op_id
    if prepare:
        if not keepdim:
            new_sizes = [self.value.sizes[i] for i in list(filter(lambda x: x not in dim, range(len(self.value.sizes))))]
        else:
            new_sizes = [(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))]
        if len(new_sizes) <= 1:
            new_value = Array(new_sizes[0], self.value.value_type)
            mean = Array(new_sizes[0], self.value.value_type)
            std = Array(new_sizes[0], self.value.value_type)
        else:
            new_value = MultiArray(new_sizes, self.value.value_type)
            mean = MultiArray(new_sizes, self.value.value_type)
            std = MultiArray(new_sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)
        
        dmean = MultiArray(self.value.sizes, self.value.value_type)
        dmean_sqr = MultiArray(self.value.sizes, self.value.value_type)
        
        new_perm = get_permute(len(self.value.sizes), dim)
        target_size = self.value.tuple_permute(self.shape, new_perm)
        input_perm = MultiArray(target_size, self.value.value_type)
        
        operation = Operation(inputs=[self.name], outputs=[output.name],
                              propagate=propagate if self.req_grad else fake_propagate, 
                              intermediate=[input_perm, mean, dmean, dmean_sqr, std], 
                              name='std')
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1

        op_id_store[op_id] = operation_id
        # op_id += 1
    if not prepare or not forward:
        operation = gradient_operation[op_id_store[op_id]]
        input = tensors[operation.inputs[0]]
        output = tensors[operation.outputs[0]]
        input_perm, mean, dmean, dmean_sqr, std = operation.intermediate
        if not forward:
            init_op_id += 1
        # pre-perm
        new_perm = get_permute(len(input.value.sizes), dim)
        input.value.permute_without_malloc(input_perm, new_perm)
        # mean
        stride = reduce(lambda x, y: x * self.value.sizes[y], dim, 1)
        summary = Array(1, input_perm.value_type)
        @for_range_opt(output.value.total_size())
        def _(i):
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += input_perm.get_vector(i*stride+j, 1)
            # summary = sum(input_perm.get_vector(i*stride, stride))
            mean.assign_vector(summary[:], i)
        summary.delete()
        break_point()    
        mean[:] *= 1 / stride
        # dmean
        @for_range_opt(output.value.total_size())
        def _(i):
            dmean_value = input_perm.get_vector(i*stride, stride) - mean.get_vector(i, 1) 
            dmean.assign_vector(dmean_value, i*stride)
        break_point()
        # var
        dmean_sqr[:] = dmean[:] ** 2
        summary = Array(1, input_perm.value_type)
        
        @for_range_opt(output.value.total_size())
        def _(i):
            summary.assign_all(0)
            @for_range(stride)
            def _(j):
                summary[:] += dmean_sqr.get_vector(i*stride+j, 1)
            # summary = sum(dmean_sqr.get_vector(i*stride, stride))
            std.assign_vector(summary[:], i)
        summary.delete()
        break_point()
        std[:] = std[:] /( stride - 1)
        # std
        std[:] = mpc_math.sqrt(std[:])
        output.value[:] = std[:]
    op_id += 1
    # record the input and output of the op
    return output

class Tensor():
    check_indices = True
    def __init__(self, value, value_type=sfix, name=None, req_grad=False, grad=None, subTensor = False):
        assert isinstance(value, Array) or isinstance(value, MultiArray) or isinstance(value, list) or isinstance(value, Tensor)
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
                dl_d[self.name] = self.grad
        tensors[self.name] = self
        self.subTensor = subTensor

    def numel(self):
        return self.value.length

    def set_req_grad(self, req_grad):
        self.req_grad = req_grad

    def reveal(self):
        return self.value.reveal()
        
    def randomize(self, *args):
        self.value.randomize(*args)
        
    def get_input_from(self, player):
        input = self.value.value_type.get_input_from(player, size=self.value.length)
        self.value.assign_vector(input)
        return self
        
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
        return self.name
    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.

    @staticmethod
    def constant(value, name=None):
        var = Tensor(value, name)
        return var

    def backward(self):
        # global prepare
        # if prepare:
        #     return 0
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
        
        # find the backward propagate chain                
        searchset = {}
        searchset[self.name] = True
        for i in range(0, index):
            op = gradient_operation[index-i-1]
            flag = False
            for it in op.outputs:
                flag = flag | searchset.get(it, False)
            if not flag:
                continue
            for it in op.inputs:
                searchset[it] = True
        
        # show tensor graph
        nodes_op = []
        nodes_tensor = []
        edges = []
        
        for i in range(0, index):
            op = gradient_operation[index-i-1]
            
            o = op.name + str(i)
                
            if index-i-1 not in nodes_op:
                nodes_op.append(o)
            
            for u in op.inputs:
                # u = u + '-' + str(tensors[u].value.value_type)[-6:-2]
                if u not in nodes_tensor:
                    nodes_tensor.append(u)
                edges.append((u, o))
            for v in op.outputs:
                # v = v + '-' + str(tensors[v].value.value_type)[-6:-2]
                if v not in nodes_tensor:
                    nodes_tensor.append(v)
                edges.append((o, v))
        
        # graph_visualization.draw_computingGraph(nodes_op, nodes_tensor, edges)
        
        
        # do backward propagate          
        for i in range(0, index):
            op = gradient_operation[index-i-1]
            dl_doutputs = gather_grad(op.outputs)
            flag = False
            for it in op.outputs:
                flag = flag | searchset.get(it, False)
            if flag:
                op.propagate(dl_doutputs, op)
        break_point()
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
        if isinstance(index, tuple):
            new_sizes = []
            tmp_new_sizes = []
            index_tuples = []
            for i in range(len(self.sizes)):
                if i < len(index):
                    if isinstance(index[i], int):
                        tmp_new_sizes.append(1)
                        index_tuples.append((index[i], 0))
                        continue
                    if isinstance(index[i], slice):
                            index_i = index[i]
                            start =  0  if index_i.start == None else index_i.start
                            end = self.sizes[i]  if index_i.stop == None else index_i.stop
                            step = 1  if index_i.step == None else index_i.step
                            size = math.ceil((end-start)/step)
                            new_sizes.append(size)
                            tmp_new_sizes.append(size)
                            index_tuples.append((start, step))
                else:
                    new_sizes.append(self.sizes[i])
                    tmp_new_sizes.append(self.sizes[i])
                    index_tuples.append(0, 1)
            new_grad = None
            if len(tmp_new_sizes) > 1: 
                res_value = MultiArray(tmp_new_sizes, self.value.value_type)
            else:
                res_value = Array(tmp_new_sizes[0], self.value.value_type)
            if self.req_grad:
                new_grad = res_value.same_shape()
            @for_range(res_value.total_size())
            def _(i):
                index_store = []
                new_index = []
                def mul(x, y):
                    return x*y
                tmp_i = i
                for j in range(len(res_value.sizes)-1):
                    left_size = (reduce(mul, res_value.sizes[j+1:]))
                    tmp_index = tmp_i// left_size
                    index_store.append(tmp_index)
                    new_index.append(index_tuples[j][0] + tmp_index * index_tuples[j][1])
                    tmp_i = tmp_i%left_size
                index_store.append(tmp_i)
                new_index.append(index_tuples[len(res_value.sizes)-1][0] + tmp_i* index_tuples[len(res_value.sizes)-1][1])
                tmp_val = self.value.get_vector_by_indices(*new_index)
                res_value.assign_vector_by_indices(tmp_val, *index_store)
                if self.req_grad:
                    tmp_val = self.grad.get_vector_by_indices(*new_index)
                    new_grad.assign_vector_by_indices(tmp_val, *index_store)
            
            if len(new_sizes) == 1:
                res_value = Array(new_sizes[0], res_value.value_type, res_value.address)
                if self.req_grad:
                    new_grad = Array(new_sizes[0], new_grad.value_type, new_grad.address)
            elif len(new_sizes) == 0:
                res_value = Array(1, res_value.value_type, res_value.address)
                if self.req_grad:
                    new_grad = Array(1, new_grad.value_type, new_grad.address)                    
            else:
                res_value.view(*new_sizes)
                if self.req_grad:
                    new_grad.view(*new_sizes)
            res = Tensor(res_value, req_grad=self.req_grad, grad=new_grad)
            return res
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
                                  address = self.value.address, index = index, debug=self.value.debug)
                if self.req_grad:
                    new_grad = \
                        MultiArray(self.sizes[1:], self.grad.value_type,
                                      address = self.grad.address, index = index, debug=self.value.debug)
                else:
                    new_grad = None
        else:
            res = self.sub_cache[key]
            return res
        res = Tensor(new_value, req_grad=self.req_grad, grad=new_grad, subTensor=True)

        self.sub_cache[key] = res
        res.check_indices = self.check_indices
        return res

    def __setitem__(self, index, other):
    #     """ Part assignment.

    #     :param index: public (regint/cint/int)
    #     :param other: container of matching size and type """
       
        if isinstance(index, tuple):
            new_sizes = []
            tmp_new_sizes = []
            index_tuples = []
            for i in range(len(self.sizes)):
                if i < len(index):
                    if isinstance(index[i], int):
                        tmp_new_sizes.append(1)
                        index_tuples.append((index[i], 0))
                        continue
                    if isinstance(index[i], slice):
                            index_i = index[i]
                            start =  0  if index_i.start == None else index_i.start
                            end = self.sizes[i]  if index_i.stop == None else index_i.stop
                            step = 1  if index_i.step == None else index_i.step
                            size = math.ceil((end-start)/step)
                            new_sizes.append(size)
                            tmp_new_sizes.append(size)
                            index_tuples.append((start, step))
                else:
                    new_sizes.append(self.sizes[i])
                    tmp_new_sizes.append(self.sizes[i])
                    index_tuples.append(0, 1)
            if isinstance(other, Tensor):
                other_value = other.value
            elif isinstance(other, MultiArray) or isinstance(other, Array):
                other_value = other
            else:
                raise CompilerError("when index is tuple, the type of other should be Tensor, MultiArray or Array")
            
            if len(tmp_new_sizes) > 1: 
                tmp_value = MultiArray(tmp_new_sizes, cint)
                tmp_res_value = MultiArray(tmp_new_sizes, predict_value_type(tmp_value, other_value))
            else:
                tmp_value = Array(tmp_new_sizes[0], cint)
                tmp_res_value = Array(tmp_new_sizes[0], predict_value_type(tmp_value, other_value))
                
            tmp_value.assign_all(1)
            boardcasted_multiarray_mul(tmp_value, other_value, tmp_res_value)
            @for_range(tmp_value.total_size())
            def _(i):
                index_store = []
                new_index = []
                def mul(x, y):
                    return x*y
                tmp_i = i
                for j in range(len(tmp_value.sizes)-1):
                    left_size = (reduce(mul, tmp_value.sizes[j+1:]))
                    tmp_index = tmp_i// left_size
                    index_store.append(tmp_index)
                    new_index.append(index_tuples[j][0] + tmp_index * index_tuples[j][1])
                    tmp_i = tmp_i%left_size
                index_store.append(tmp_i)
                new_index.append(index_tuples[len(tmp_value.sizes)-1][0] + tmp_i* index_tuples[len(tmp_value.sizes)-1][1])
                tmp_val = tmp_res_value.get_vector_by_indices(*index_store)
                self.value.assign_vector_by_indices(tmp_val, *new_index)
            tmp_value.delete()
            tmp_res_value.delete()
            return
        if isinstance(other, Tensor):
            self.value[index] = other.value
        else:
            self.value[index] = other

                    
    @buildingblock("masked_fill_")
    def masked_fill_(self, mask, value):
        b =  mask *value 
        return self + b

    @staticmethod
    def ones(*sizes, value_type = sfix, req_grad = False):
        if len(sizes) == 1 and isinstance(sizes[0], list):
            sizes = sizes[0]
        else:
            sizes = list(sizes)
        if len(sizes) == 0 or value_type is None:
            raise CompilerError("the shape of a tensor must be a not-null list and value type must be determined")
        if len(sizes) == 1:
            res_value = Array(sizes[0], value_type)
        if len(sizes) > 1:
            res_value = MultiArray(sizes, value_type)
        res_value.assign_all(1)
        res = Tensor(res_value, req_grad=req_grad)        
        return res

    @staticmethod
    def zeros(*sizes, value_type = sfix, req_grad = False):
 
        if len(sizes) == 1 and isinstance(sizes[0], list):
            sizes = sizes[0]
        else:
            sizes = list(sizes)        
        if len(sizes) == 0 or value_type is None:
            raise CompilerError("the shape of a tensor must be a not-null list and value type must be determined")
        if len(sizes) == 1:
            res_value = Array(sizes[0], value_type)
        if len(sizes) > 1:
            res_value = MultiArray(sizes, value_type)
        res_value.assign_all(0)
        res = Tensor(res_value, req_grad=req_grad)        
        return res

    @staticmethod
    def eye(m, n =None, value_type = sfix, req_grad = False):
        assert n is None or m == n
        res_value = MultiArray([m, m], value_type)
        @for_range(m)
        def _(i):
            res_value[i][i] = 1
        res = Tensor(res_value, req_grad=req_grad)    
        return res
    
    @staticmethod
    def arange(start , end, step =1, value_type = sfix, req_grad = False):
        size = math.ceil((end-start)/step)
        res_value = Array(size, value_type)
        @for_range(start, end, step)
        def _(i):
            res_value.assign_vector(value_type(i), i/step)
        res = Tensor(res_value, req_grad=req_grad)  
        return res
    
    @staticmethod
    def triu(input, k = 0, inplace = False):
        assert len(input.sizes) >= 2
        if inplace:
            res = input
        else:
            res_value = input.value.same_shape()
            res_value[:] = input.value[:]
            res = Tensor(res_value, req_grad=input.req_grad)
        mask_value = MultiArray([input.sizes[-2],input.sizes[-1]], value_type=cint)
        mask_value.assign_all(1)
        length = min(input.sizes[-2],input.sizes[-1])
        if k<= 0:
            length = length + k -1
            if length <= 0:
                return res
            @for_range(length)
            def _(i):
                @for_range(length - i)
                def _(j):
                    mask_value.assign_vector_by_indices(cint(0), input.sizes[-2] - i -1, j)
        else:
            if k >= input.sizes[-1]:
                res.value.assign_all(0)
                return res
            @for_range(length)
            def _(i):
                @for_range(i+k)
                def _(j):
                    mask_value.assign_vector_by_indices(cint(0), i, j)             
        boardcasted_multiarray_mul(res.value, mask_value, res.value)       
        return res
    
    
    @staticmethod
    def tril(input, k = 0, inplace = False):
        assert len(input.sizes) >= 2
        if inplace:
            res = input
        else:
            res_value = input.value.same_shape()
            res_value[:] = input.value[:]
            res = Tensor(res_value, req_grad=input.req_grad)
        mask_value = MultiArray([input.sizes[-2],input.sizes[-1]], value_type=cint)
        mask_value.assign_all(1)
        length = min(input.sizes[-2],input.sizes[-1])
        if k >= 0:
            length = length - k -1
            if length <= 0:
                return res
            @for_range(length)
            def _(i):
                @for_range(i-k)
                def _(j):
                    mask_value.assign_vector_by_indices(cint(0), i, input.sizes[-1] - j-1)
        else:
            if -k >= input.sizes[-2]:
                res.value.assign_all(0)
                return res
            @for_range(length)
            def _(i):
                @for_range(i-k)
                def _(j):
                    mask_value.assign_vector_by_indices(cint(0), j, i)
            boardcasted_multiarray_mul(res.value, mask_value, res.value)       
        return res    
    
    def argmax(self, dim, keepdim=False):
        dim = [dim]
        if not keepdim:
            new_sizes = [self.value.sizes[i] for i in list(filter(lambda x: x not in dim, range(len(self.value.sizes))))]
        else:
            new_sizes = [(1 if i in dim else self.value.sizes[i]) for i in range(len(self.value.sizes))]
        if len(new_sizes) <= 1:
            new_value = Array(new_sizes[0], self.value.value_type)
        else:
            new_value = MultiArray(new_sizes, self.value.value_type)
        output = Tensor(new_value, req_grad=self.req_grad)

        new_perm = get_permute(len(self.value.sizes), dim)
        target_size = self.value.tuple_permute(self.value.sizes, new_perm)
        temp = MultiArray(target_size, self.value.value_type)
        self.value.permute_without_malloc(temp, new_perm)
        
        stride = self.value.sizes[dim[0]]
        @for_range_opt(self.value.total_size()//stride)
        def _(i):
            t = temp.get_vector(i*stride, stride)
            output.value.assign_vector(argmax(t), i)
        temp.delete()
        return output
    
    @buildingblock("mv-forward")
    def mv(self, other,out=None):
        # mul of Two-dimension * Array,return an output,whose type is Tensor and value is Array
        @backwardbuildingblock(get_program().globalbuildingblock[:-11]+"-mv-backward")
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
        global init_op_id
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
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input1 = tensors[inputs[0]]
            input2 = tensors[inputs[1]]
            output = tensors[outputs[0]]
            input1.value.mv(input2.value,output.value)
            if not forward:
                init_op_id += 1
            # n_threads=10 if input1.shape[0]>=1000 else 1
            # @multithread(n_threads,input1.shape[0])
            # def _(base, size):
            #     output.value.assign_part_vector(input1.value.direct_mul(input2.value,indices=(regint.inc(size,base=base),regint.inc(input1.shape[1]), regint.inc(input2.shape[0]),regint.inc(1))),base)
        op_id += 1# record the input and output of the op
        return output
    
    @buildingblock("mm-forward")
    def mm(self, other): #Two-dimension * two-dimension,return an output,whose type is Tensor.
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-11]+"-mm-backward")
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
        global init_op_id
        if prepare:
            assert self.value.value_type==other.value.value_type,"Invalid Data Type"
            assert len(self.shape)==len(other.shape)==2 and self.shape[1]==other.shape[0],"Invalid Dimension"
            new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad or other.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input1 = tensors[inputs[0]]
            input2 = tensors[inputs[1]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            input1.value.mm(input2.value, output.value)
        op_id += 1  # record the input and output of the op
        return output
    
    @buildingblock("singlebmm-forward")
    def single_bmm(self, other: 'Tensor') -> 'Tensor':
        '''
        Performs a batch matrix-matrix product of matrices stored in self and other.
        Note: This function does not broadcast
        :param self.shape: [*b, n, m]
        :param other.shape: [m, p]
        :return: return.shape: [*b, n, p]
        '''
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-18]+"-singlebmm-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            input1, input2 = tensors[operation.inputs[0]], tensors[operation.inputs[1]]
            if self.req_grad:
                cur_dinput1 = dl_dy.single_bmm_trans_to(input2.value)
                dl_d[operation.inputs[0]][:] += cur_dinput1[:]
                cur_dinput1.delete()
            if other.req_grad:
                cur_dinput2 = input1.value.trans_bmm_to(dl_dy, is_reduce=True)
                dl_d[operation.inputs[1]][:] += cur_dinput2[:]
                cur_dinput2.delete()
        # forward
        global op_id
        global init_op_id
        if prepare:
            assert len(self.sizes) >= 3 and self.sizes[-1] == other.sizes[0], "Invalid Dimension"
            batch, n, p = self.sizes[:-2], self.sizes[-2], other.sizes[-1]
            output = Tensor(MultiArray([*batch, n, p], other.value.value_type), req_grad=self.req_grad or other.req_grad)
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad or other.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs, outputs = operation.inputs, operation.outputs
            input1, input2, output = tensors[inputs[0]], tensors[inputs[1]], tensors[outputs[0]]
            input1.value.single_bmm(input2.value, output.value)
            if not forward:
                init_op_id += 1
        op_id += 1
        return output
    
    
    @buildingblock("bmm-forward")
    def bmm(self, other: 'Tensor') -> 'Tensor':
        '''
        Performs a batch matrix-matrix product of matrices stored in self and other.
        Note: This function does not broadcast
        :param self.shape: [*b, n, m]
        :param other.shape: [*b, m, p]
        :return: return.shape: [*b, n, p]
        '''
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-bmm-backward")
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
        global init_op_id
        if prepare:
            assert len(self.sizes) == len(other.sizes) >= 3 and self.sizes[-1] == other.sizes[-2], "Invalid Dimension"
            batch, n, p = self.sizes[:-2], self.sizes[-2], other.sizes[-1]
            output = Tensor(MultiArray([*batch, n, p], other.value.value_type), req_grad=self.req_grad or other.req_grad)
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], 
                                  propagate=propagate if self.req_grad or other.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs, outputs = operation.inputs, operation.outputs
            input1, input2, output = tensors[inputs[0]], tensors[inputs[1]], tensors[outputs[0]]
            input1.value.bmm(input2.value, output.value)
            if not forward:
                init_op_id += 1
        op_id += 1
        return output

    @buildingblock("dot-forward")
    def dot(self, other):
        #Mul of two Array 
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-dot-backward")
        def propagate(dl_doutputs, operation):
            dl_dy,=dl_doutputs
            if self.req_grad:
                dl_d[operation.inputs[0]][:]+= tensors[operation.inputs[1]].value[:]*dl_dy #dA=dC*B+dA
            if other.req_grad:
                dl_d[operation.inputs[1]][:]+= tensors[operation.inputs[0]].value[:]*dl_dy #dB=dC*A+dB
        global op_id
        global init_op_id
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
            # op_id += 1
        if not prepare or not forward:
            operation=gradient_operation[op_id_store[op_id]]
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            output=tensors[operation.outputs[0]]
            if not forward:
                init_op_id += 1
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
        # return element_wise_sub(self, other)
        return element_wise_add(self, -other)

    def __neg__(self):
        return ops_mul_constant(self, -1)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ops_mul_constant(self, 1./other)
        return element_wise_div(self, other)

    def __len__(self):
        return len(self.value)

    @buildingblock("view-forward")
    def view(self, *sizes):
        
        @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-view-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        global init_op_id
        if prepare:
            product = reduce(lambda x, y: x*y, self.shape)
            if len(sizes) == 1 and isinstance(sizes[0], int):
                sizes = sizes[0]
                assert sizes == product, "Invalid Dimension"
                new_value = Array(sizes, self.value.value_type)
            else:
                if len(sizes) == 1 and isinstance(sizes[0], list):
                    sizes = sizes[0]
                elif len(sizes) == 1 and isinstance(sizes[0], tuple):
                    sizes = list(sizes[0])
                else:
                    sizes = list(sizes)
                assert all(isinstance(x, int) and x > -2 for x in sizes), "Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1) == 1, "-1 Occurs More than Once "
                    tmp = reduce(lambda x, y: x*y, sizes)
                    assert product % (-tmp) == 0, "Invalid Dimension"
                    sizes[sizes.index(-1)] = int(product/(-tmp))
                new_value = MultiArray(sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value.assign(self.value)
        op_id += 1
        return output

    def flatten(self, start_dim, end_dim=-1):
        sizes = self.sizes
        length = len(sizes)
        if start_dim < 0:
            start_dim = length + start_dim
        
        if end_dim < 0:
            end_dim = length + end_dim 
        
        assert start_dim >= 0 and end_dim >= 0 and start_dim < length and end_dim < length and start_dim < end_dim
        new_sizes = []
        new_len = 1
        for i in range(length):
            if i < start_dim or i > end_dim:
                new_sizes.append(sizes[i])  
                continue
            if i == end_dim:
                new_len *= sizes[i]
                new_sizes.append(int(new_len))
                continue
            new_len *= sizes[i]
        return self.view(new_sizes)
                    
    @buildingblock("squeeze-forward")
    def squeeze(self, dim=None):
        
        @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-squeeze-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        global init_op_id
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
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad or other.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            output.value.assign(self.value)
            if not forward:
                init_op_id += 1
        op_id += 1
        return output

    @buildingblock("unsqueeze-forward")
    def unsqueeze(self, dim):
        
        @backwardbuildingblock(get_program().globalbuildingblock[:-18]+"-unsqueeze-backward")
        def propagate(dl_doutputs, operation):
            dl_d[operation.inputs[0]].assign(dl_doutputs[0])
        global op_id
        global init_op_id
        if prepare:
            new_sizes = list(self.shape)
            assert isinstance(dim, int) and dim <= len(self.shape) and dim >= -len(self.shape), "Invalid Dimension"
            new_sizes.insert(dim, 1)
            new_value = MultiArray(new_sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value.assign(self.value)
        op_id += 1
        return output

    @buildingblock("gather-forward")
    def gather(self, dim, index):
        # todo
        @backwardbuildingblock(get_program().globalbuildingblock[:-15]+"-gather-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_dself = dl_d[operation.inputs[0]]
            @for_range(dl_dy.total_size())
            def _(i):
                index_store = []
                new_index = []
                def mul(x, y):
                    return x*y
                tmp_i = i
                for j in range(len(dl_dy.sizes)-1):
                    left_size = (reduce(mul, dl_dy.sizes[j+1:]))
                    tmp_index = tmp_i// left_size
                    index_store.append(tmp_index)
                    new_index.append(tmp_index)
                    tmp_i = tmp_i%left_size
                index_store.append(tmp_i)
                new_index.append(tmp_i)
                new_index[dim] = index.value.get_vector_by_indices(*index_store)
                tmp_val = dl_dy.get_vector_by_indices(*index_store)
                dl_dself.assign_vector_by_indices(dl_dself.get_vector_by_indices(*new_index)+tmp_val, *new_index)            
               
        global op_id
        global init_op_id
        if prepare:
            assert len(self.sizes) == len(index.sizes)
            assert index.value.value_type == cint or index.value.value_type == regint
            for i in range(len(self.sizes)):
                if i!=dim and index.sizes[i] > self.sizes[i]:
                    raise CompilerError("wrong dimension of index in gather function")
            if len(index.sizes) == 0:
                new_value = Array(index.sizes, self.value.value_type)
            else:
                new_value = MultiArray(index.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            @for_range(output.value.total_size())
            def _(i):
                index_store = []
                new_index = []
                def mul(x, y):
                    return x*y
                tmp_i = i
                for j in range(len(output.sizes)-1):
                    left_size = (reduce(mul, output.sizes[j+1:]))
                    tmp_index = tmp_i// left_size
                    index_store.append(tmp_index)
                    new_index.append(tmp_index)
                    tmp_i = tmp_i%left_size
                index_store.append(tmp_i)
                new_index.append(tmp_i)
                new_index[dim] = index.value.get_vector_by_indices(*index_store)
                tmp_val = self.value.get_vector_by_indices(*new_index)
                output.value.assign_vector_by_indices(tmp_val, *index_store)
        op_id += 1
        return output

    @buildingblock("reshape-forward")
    def reshape(self, *sizes):
        
        @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-reshape-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        global init_op_id
        if prepare:
            product = reduce(lambda x, y: x*y, self.shape)
            if len(sizes) == 1 and isinstance(sizes[0], int):
                sizes = sizes[0]
                assert sizes == product, "Invalid Dimension"
                new_value = Array(sizes, self.value.value_type)
            else:
                if len(sizes) == 1 and isinstance(sizes[0], list):
                    sizes = sizes[0]
                elif len(sizes) == 1 and isinstance(sizes[0], tuple):
                    sizes = list(sizes[0])
                else:
                    sizes = list(sizes)
                assert all(isinstance(x, int) for x in sizes), "Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1) == 1, "-1 Occurs More than Once "
                    tmp = reduce(lambda x, y: x*y, sizes)
                    assert product % (-tmp) == 0, "Invalid Dimension"
                    sizes[sizes.index(-1)] = int(product/(-tmp))
                new_value = MultiArray(sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            if self.req_grad:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value.assign(self.value)
        op_id += 1
        return output



    @buildingblock("permute-forward")
    def permute(self, *new_perm):  # todo :这里的参数不应该是list类型的new-perm，而应该是*newperm :pytorch中：x.permute(2, 0, 1)
        if not isinstance(new_perm[0], list):
            new_perm = list(new_perm)
        else:
            new_perm = new_perm[0]
        @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-permute-backward")
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            L=len(self.shape)
            inv_new_perm=[None]*L
            for i in range(L):
                inv_new_perm[new_perm[i]]=i #s2[s1[i]]=i
            dl_dy.permute_without_malloc(dl_d[operation.inputs[0]],inv_new_perm)
        global op_id
        global init_op_id
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
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            outputs = operation.outputs
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            self.value.permute_without_malloc(output.value, new_perm)  # output的值在参数中传入后被修改
        op_id += 1
        return output

    @buildingblock("transpose-forward")
    def transpose(self, *indexs):
        if len(indexs)>0:
            indexs = list(indexs)
            new_index = []
            for i in range(len(self.sizes)):
                if i == indexs[0]%len(self.sizes):
                    new_index.append(indexs[1]%len(self.sizes))
                    continue
                if i == indexs[1]%len(self.sizes):
                    new_index.append(indexs[0]%len(self.sizes))
                    continue
                new_index.append(i)
            return self.permute(*new_index)
        @backwardbuildingblock(get_program().globalbuildingblock[:-18]+ "-transpose-backward")
        def propagate(dl_doutputs, operation):
            if isinstance(dl_doutputs[0], Array):
                dl_d[operation.inputs[0]][:] += dl_doutputs[0][:]
            else:
                dl_d[operation.inputs[0]][:] += dl_doutputs[0].transpose()[:]
        global op_id
        global init_op_id
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
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            input = tensors[operation.inputs[0]]
            output = tensors[operation.outputs[0]]
            if not forward:
                init_op_id += 1
            if len(self.shape) == 1:  # in this case:Array
                output.value[:] = input.value[:]
            else:
                output.value = input.value.transpose()
        op_id += 1
        return output

    @buildingblock("concat-forward")
    def concat(self, other, dim=0):  # 按照dim指定维度进行拼接
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            size_pre=reduce(lambda x,y:x*y,input1.shape[dim:])
            size_next=reduce(lambda x,y:x*y,input2.shape[dim:]) 
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
        global init_op_id
        if prepare:
            assert self.value.value_type is other.value.value_type, "Invalid value_type"
            if isinstance(self.value, Array) and isinstance(other.value, Array):
                target_len = self.value.length + other.value.length
                new_value = Array(target_len, self.value.value_type)
            else:
                assert len(self.shape)==len(other.shape),"Inequal Dimension"
                for i in range(len(self.shape)):
                    if i != dim and self.shape[i] != other.shape[i]:
                        raise ValueError("Invalid Dimension")
                target_size = other.value.shape
                target_size[dim] += self.value.shape[dim]
                new_value = MultiArray(target_size, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad or other.req_grad)
            if self.req_grad or other.req_grad:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
            else:
                operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation=gradient_operation[op_id_store[op_id]]
            size_pre=reduce(lambda x,y:x*y,self.shape[dim:])
            size_next=reduce(lambda x,y:x*y,other.shape[dim:])
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            output=tensors[operation.outputs[0]]
            index=regint(0)    
            if not forward:
                init_op_id += 1
            @for_range(input1.value.length//size_pre)
            def _(i):  
                #can not convert this to @for_range for the error info of "local variable 'index' referenced before assignment"
                output.value.assign_vector(input1.value.get_vector(i*size_pre,size_pre),index)
                index.update(index+size_pre)
                output.value.assign_vector(input2.value.get_vector(i*size_next,size_next),index)
                index.update(index+size_next)
        op_id+=1
        return output
    
    @buildingblock("chunk")
    def chunk(self, chunks, dim=0):
        stride = reduce(lambda x,y:x*y,self.shape[dim+1:])
        prefix_total = self.value.total_size() // stride // self.shape[dim]
        new_dim_size = (self.sizes[dim]+chunks-1) // chunks
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            input=tensors[operation.inputs[0]]
            dl_dy = [dld for dld in dl_doutputs]
            @for_range(prefix_total)
            def _(i):
                for j in range(self.sizes[dim]):
                    dim_size = dl_dy[j//new_dim_size].sizes[dim]
                    v = dl_dy[j//new_dim_size].get_vector(i*dim_size*stride+(j%dim_size)*stride,stride)
                    input.grad.assign_vector(v,i*self.sizes[dim]*stride+j*stride)
            return             
        global op_id
        global init_op_id
        if prepare:

            

            new_chunks = (self.sizes[dim]-1)// new_dim_size
            
            new_size = self.sizes[:dim] + (new_dim_size,) + self.sizes[dim+1:]
            new_size_last = self.sizes[:dim] + (self.sizes[dim] - (new_chunks) * new_dim_size,) + self.sizes[dim+1:]
            
            # print(new_dim_size)
            # print(new_chunks)
            # print(new_size)
            # print(new_size_last)
            # print(stride)
            # print(prefix_total)
            
            output = [Tensor(MultiArray(new_size, sfix), req_grad=self.req_grad) for i in range(new_chunks)]
            output.append(Tensor(MultiArray(new_size_last, sfix), req_grad=self.req_grad))

            operation = Operation(
                inputs=[self.name], outputs=[out.name for out in output], 
                propagate=propagate if self.req_grad else fake_propagate, )
            
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
        if not prepare or not forward:
            operation=gradient_operation[op_id_store[op_id]]
            input=tensors[operation.inputs[0]]
            output=[tensors[operation.outputs[i]] for i in range(len(operation.outputs))]
            @for_range(prefix_total)
            def _(i):
                for j in range(self.sizes[dim]):
                    v = self.value.get_vector(i*self.sizes[dim]*stride+j*stride,stride)
                    dim_size = output[j//new_dim_size].value.sizes[dim]
                    output[j//new_dim_size].value.assign_vector(v,i*dim_size*stride+(j%dim_size)*stride)
        op_id+=1
        return output
    
    @buildingblock("expand")
    def expand(self, sizes):
        for i in range(len(sizes)):
            if sizes[i] == -1:
                sizes[i] = self.value.sizes[i]
        temp_value = MultiArray(sizes, cint)
        temp = Tensor(temp_value, req_grad=False)
        return self + temp
    
    @buildingblock("expand_as")
    def expand_as(self, other):
        return self.expand(list(other.value.sizes))

    @buildingblock("repeat")
    def repeat(self, *sizes):
        sizes = list(sizes)
        start = 0
        if len(sizes) > len(self.value.sizes):
            new_sizes = sizes[:len(sizes)-len(self.value.sizes)]
            start = len(sizes)-len(self.value.sizes)
        else:
            new_sizes = []
        for i in range(len(self.value.sizes)):
            new_sizes.append(self.value.sizes[i] * sizes[start+i])
        return self.expand(new_sizes)
    
    @buildingblock("gt")
    def gt(input, other):
        if isinstance(other, int) or isinstance(other, float):
            new_value = input.value.same_shape()
            output = Tensor(new_value, req_grad=input.req_grad)
            output.value[:] = output.value[:] > other
            return output
        if isinstance(input.value, MultiArray) or isinstance(other.value, MultiArray):
            if input.value.total_size()>other.value.total_size():
                new_value = MultiArray(input.value.sizes, predict_value_type(input.value, other.value))
            else:
                new_value = MultiArray(other.value.sizes, predict_value_type(input.value, other.value))
        else:
            if input.value.total_size()>other.value.total_size():
                new_value = Array(input.value.sizes[0], predict_value_type(input.value, other.value))
            else:
                new_value = Array(other.value.sizes[0], predict_value_type(input.value, other.value))
        output = Tensor(new_value, req_grad=input.req_grad or other.req_grad)
        
        boardcasted_multiarray_sub(input.value, other.value, output.value)
        
        output.value[:] = output.value[:] >0
        return output
        
    @buildingblock("abs-forward")
    def abs(self):
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
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name], 
                                  propagate=propagate if self.req_grad else fake_propagate, 
                                  intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            larger = input.value[:] > 0
            less = input.value[:]<0
            final=larger-less
            operation.intermediate[0].assign_vector(final)  # write to mem
            if not forward:
                init_op_id += 1
            output.value[:] = final * input.value[:]
        op_id += 1
        # record the input and output of the op
        return output

    @buildingblock("exp-forward")
    def exp(self):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-exp-backward")
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
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name], 
                                  propagate=propagate if self.req_grad else fake_propagate, 
                                  intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            ex = mpc_math.pow_fx(math.e, input.value[:])
            operation.intermediate[0].assign_vector(ex)

            output.value[:] = ex
        op_id += 1
        # record the input and output of the op
        return output

    @buildingblock("log-forward")
    def log(self, base=math.e):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-log-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += 1 / (self.value[:] * np.log(base)) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value[:] = mpc_math.log_fx(input.value[:], base)

        op_id += 1
        # record the input and output of the op
        return output

    @buildingblock("pow-forward")
    def pow(self, pow):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-pow-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += pow * mpc_math.pow_fx(self.value[:], pow-1) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value[:] = mpc_math.pow_fx(input.value[:], pow)

        op_id += 1
        # record the input and output of the op
        return output

    # todo: unsing 1 / mpc_math.sqrt(x) when 'approx' is False
    @buildingblock("invsqrt-forward")
    def invsqrt(self, eps=1e-12):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-invsqrt-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            outputs = operation.outputs
            output = tensors[outputs[0]]
            dl_dself = dl_d[inputs[0]]
        
            dl_dself[:] += -0.5 * output.value[:] * output.value[:] * output.value[:] * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate,
                                  name='invsqrt')
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1

            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            # output.value[:] = mpc_math.InvertSqrt(input.value[:] + eps)
            @multithread(1, output.value.total_size())
            def _(base, size):
                output.value.assign_vector(mpc_math.InvertSqrt(input.value.get_vector(base, size)+eps) , base)

        op_id += 1
        # record the input and output of the op
        return output
    
    @buildingblock("cos-forward")
    def cos(self):
        
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-cos-backward")
        def propagate(dl_doutputs, operation):  # dl_outputs is Tensor.value
            dl_dx, = dl_doutputs
            dl_dself = dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*(-mpc_math.sin(self.value[:]))
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):  # Array is instance of tensor?
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name], 
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]  # input is Tensor
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value[:] = mpc_math.cos(input.value[:])
        op_id += 1
        return output

    @buildingblock("sin-forward")
    def sin(self):
        @backwardbuildingblock(get_program().globalbuildingblock[:-12]+"-sin-backward")
        def propagate(dl_doutputs, operation):  # dl_outputs is Tensor.value
            dl_dx, = dl_doutputs
            dl_dself = dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*mpc_math.cos(self.value[:])
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):  # Array is instance of tensor?
                new_value = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation)-1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            input = tensors[inputs[0]]  # input is Tensor
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            output.value[:] = mpc_math.sin(input.value[:])
        op_id += 1
        return output

    def mean(self, dim=None, keepdim=False):
        if isinstance(dim, int):
            dim=[dim]
        if isinstance(self.value, Array) or dim==None:
            return mean_of_array(self)
        else:
            return mean_of_multiarray(self, dim, keepdim)

    def sum(self, dim=None, keepdim=False):
        if isinstance(dim, int):
            dim=[dim]
        if isinstance(self.value, Array) or dim==None:
            return sum_of_array(self)
        else:
            return sum_of_multiarray(self, dim, keepdim)

    def std(self, dim=None, keepdim=False):
        if isinstance(dim, int):
            dim=[dim]
        if isinstance(self.value, Array) or dim==None:
            return std_of_array(self)
        else:
            return std_of_multiarray(self, dim, keepdim)

    def var(self, dim=None, keepdim=False, unbiased=False):
        if isinstance(dim, int):
            dim=[dim]
        if isinstance(self.value, Array) or dim==None:
            return var_of_array(self, unbiased)
        else:
            return var_of_multiarray(self, dim, keepdim, unbiased)

    def norm(self, dim=None, keepdim=False):
        pass

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
                
                changed_output_1.permute_without_malloc(output.value,get_permute_back(len(output.sizes), [ dim%len(output.sizes) ]))
        set_opid(op_id+1)  # record the input and output of the op
        return output
        
    @buildingblock("relu-forward")
    def relu(self, inplace=False):  
        # Considering that the saved memory overhead has very little impact on MPC computing performance, 
        #the inplace parameter is not considered
        op_id = get_opid()
        @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-relu-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            dl_d[self.name]+=operation.intermediate[0][:]*dl_dy[:]        
        prepare = get_prepare()
        global init_op_id
        if prepare:
            assert isinstance(self, Tensor),"Invalid Input"
            if isinstance(self.value,Array):
                new_value=Array(self.shape[0],self.value.value_type)
                inter=Array(self.shape[0],sint)
            else:
                new_value=MultiArray(list(self.shape) ,self.value.value_type)
                inter=MultiArray(list(self.shape) ,sint)
            output = Tensor(new_value, req_grad=self.req_grad)
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate,
                                  intermediate=[inter])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # set_opid(op_id+1)
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            output = tensors[operation.outputs[0]]
            if not forward:
                init_op_id += 1
            @multithread(1, 1)
            def _(base, size):
                larger=0 < input.value[:]
                operation.intermediate[0].assign_vector(larger)
                output.value[:] = (larger).if_else(input.value[:], 0) 
        set_opid(op_id+1)  # record the input and output of the op
        return output
        
    @buildingblock("sigmoid-forward")
    def sigmoid(self,approx=False): # added approx parameter to speed up the computation
        op_id = get_opid()
        @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-sigmoid-backward")
        def propagate(dl_doutputs, operation):
            dl_dy, = dl_doutputs
            input_ = tensors[operation.inputs[0]]
            output = tensors[operation.outputs[0]]
            # if input_.req_grad:
            dl_d[input_.name]+=output.value[:]*(1-output.value[:])*dl_dy[:]
                
        prepare = get_prepare()
        global init_op_id
        if prepare:
            assert isinstance(self, Tensor),"Invalid Input"
            if isinstance(self.value,Array):
                new_value=Array(self.shape[0],self.value.value_type)
            else:
                new_value=MultiArray(list(self.shape) ,self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            
            operation = Operation(inputs=[self.name], outputs=[output.name], 
                                  propagate=propagate if self.req_grad else fake_propagate)
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # set_opid(op_id+1)
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            output = tensors[operation.outputs[0]]
            if not forward:
                init_op_id += 1
            if approx:
                output.value[:]= approx_sigmoid(self.value[:])
            else:
                output.value[:] =  sigmoid_from_e_x(self.value[:],exp(-self.value[:]))
        set_opid(op_id+1)  # record the input and output of the op
        return output

    @buildingblock("tanh-forward")
    def tanh(self):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-13]+"-tanh-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]
            # denominator = inter[:] * inter[:] + 2 * inter[:] + 1
            # dl_dself[:] += 4 * inter[:] / denominator * dl_dx[:]
            dl_dself[:] += (1 - inter[:] * inter[:])* dl_dx[:]
            # dl_dself[:] += s.if_else(0, t.if_else(0, (1 - inter[:] * inter[:]))) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
                s = Array(self.value.length, self.value.value_type)
                t = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
                s = MultiArray(self.value.sizes, self.value.value_type)
                t = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate,
                                  intermediate=[inter, s, t])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            inter, s, t = operation.intermediate
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1
            # input.print_reveal_nested()
            tmp_input = 2 * input.value[:]
            
            ltz = tmp_input < 0
            sign = 1- 2 *  ltz
            tmp_input = tmp_input *sign
            
            ex =  mpc_math.exp_fx(-tmp_input, 8)

            tmp_input = ex+1
            cfix.div_iters = 2
            cfix.all_pos = True
            cfix.div_initial = 0.75
            tmp_input = 1 / tmp_input
            cfix.div_iters = 10
            cfix.all_pos = False
            cfix.div_initial = None
            tmp_input = tmp_input + ltz  - 2*tmp_input*ltz
            output.value[:] =  tmp_input *2 -1
            
            # limit = math.log(2 ** (input.value[:].k - input.value[:].f - 2)) / 2
            # s = input.value[:] < -limit
            # t = input.value[:] > limit
            # output.value[:] = s.if_else(-1, t.if_else(1, output.value[:]))
            inter.assign_vector(output.value[:])
        op_id += 1
        # record the input and output of the op
        return output
    
    @buildingblock("Hardtanh-forward")
    def Hardtanh(self, min_val=-1.0, max_val=1.0):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-17]+"-Hardtanh-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter, s, t = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]

            dl_dself[:] += s.if_else(0, t.if_else(0, 1)) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
                s = Array(self.value.length, self.value.value_type)
                t = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
                s = MultiArray(self.value.sizes, self.value.value_type)
                t = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate,
                                  intermediate=[inter, s, t])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            inter, s, t = operation.intermediate
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1 
            
            s = input.value[:] < min_val
            t = input.value[:] > max_val
            output.value[:] = s.if_else(min_val, t.if_else(max_val, input.value[:]))
            inter.assign_vector(output.value[:])
        op_id += 1
        # record the input and output of the op
        return output
    
    @buildingblock("Relu6-forward")
    def Relu6(self):
        # backward
        @backwardbuildingblock(get_program().globalbuildingblock[:-14]+"-Relu6-backward")
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter = operation.intermediate[0]  # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]

            dl_dself[:] += s.if_else(0, t.if_else(0, 1)) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        global init_op_id
        if prepare:
            if isinstance(self.value, Array):
                new_value = Array(self.value.length, self.value.value_type)
                inter = Array(self.value.length, self.value.value_type)
                s = Array(self.value.length, self.value.value_type)
                t = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                inter = MultiArray(self.value.sizes, self.value.value_type)
                s = MultiArray(self.value.sizes, self.value.value_type)
                t = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value, req_grad=self.req_grad)
            
            operation = Operation(inputs=[self.name], outputs=[output.name],
                                  propagate=propagate if self.req_grad else fake_propagate,
                                  intermediate=[inter, s, t])
            gradient_operation.append(operation)
            operation_id = len(gradient_operation) - 1
            op_id_store[op_id] = operation_id
            # op_id += 1
        if not prepare or not forward:
            operation = gradient_operation[op_id_store[op_id]]
            inputs = operation.inputs
            outputs = operation.outputs
            inter, s, t = operation.intermediate
            input = tensors[inputs[0]]
            output = tensors[outputs[0]]
            if not forward:
                init_op_id += 1 
            
            s = input.value[:] < 0
            t = input.value[:] > 6
            output.value[:] = s.if_else(0, t.if_else(6, input.value[:]))
            inter.assign_vector(output.value[:])
        op_id += 1
        # record the input and output of the op
        return output
    
    def size(self, dim = None):
        if dim == None:
            return self.value.sizes
        else:
            return self.value.sizes[dim]

    def zero_grad(self):
        if self.req_grad:
            self.grad.assign_all(0)
        
    def assign_all(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self.value.assign_all(value)

def exp_for_softmax(x):
    m = util.max(x) - get_limit(x[0]) + math.log(len(x))
    mv = m.expand_to_vector(len(x))
    try:
        x = x.get_vector()
    except AttributeError:
        x = sfix(x)
    if use_mux:
        return exp(x - mv), m
    else:
        return (x - mv > -get_limit(x)).if_else(exp(x - mv), 0)


# @vectorize
# def approx_sigmoid(x, n=5):
#     """ Piece-wise approximate sigmoid as in
#     `Hong et al. <https://arxiv.org/abs/2002.04344>`_

#     :param x: input
#     :param n: number of pieces, 3 (default) or 5
#     """
#     if n == 5:
#         cuts = [-5, -2.5, 2.5, 5]
#         le = [0] + [x <= cut for cut in cuts] + [1]
#         select = [le[i + 1] - le[i] for i in range(5)]
#         outputs = [cfix(10 ** -4),
#                    0.02776 * x + 0.145,
#                    0.17 *x + 0.5,
#                    0.02776 * x + 0.85498,
#                    cfix(1 - 10 ** -4)]
#         return sum(a * b for a, b in zip(select, outputs))
#     else:
#         a = x < -0.5
#         b = x > 0.5
#         return a.if_else(0, b.if_else(1, 0.5 + x))
    
    
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

def reset():
        global init_op_id
        untrain()
        init_op_id = 0
        reset_op_id()
        reset_gloabal_store()
# reset operation
def reset_gloabal_store():
    gradient_operation.clear()
    for key, item in tensors.items():
        if not isinstance(item.value.address, int) or item.subTensor:
            continue
        item.value.delete()
    tensors.clear()
    for key, item in dl_d.items():
        if item is not None:
            item.delete()
    dl_d.clear()
    op_id_store.clear()

# call this function after each iteration
def reset_op_id():
    global op_id
    global init_op_id
    op_id = init_op_id


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

def get_forward():
    global forward
    return forward

def get_init_op_id():
    global init_op_id
    return init_op_id

def set_init_op_id(new_id):
    global init_op_id
    init_op_id = new_id

def untrain():
    global prepare
    prepare = True

def autograd_function(func):
    global init_op_id
    def wrapper(*args, **kw):
        func(*args, **kw)
        untrain()
        init_op_id = 0
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
        @for_range_opt(n)
        def _(i):
            per_res.assign_vector(vec_softmax(per_x.get_vector(i*m, m)), index)
            index.update(index+m)
        break_point()
        per_x.view(*batch, m), per_res.view(*batch, m)
        per_res.swap_single_dim(dim, -1, res)
        return res

def vec_softmax(x):
    # e_x = exp_for_softmax(x)
    max = util.max(x)
    index = x == max
    tmp = x*x*x
    tmp = x> tmp
    e_x = mpc_math.exp_fx(x -max , 8)
    sfix.all_pos = True
    res = e_x  / sum(e_x)
    sfix.all_pos = False
    return res
    

# def broadcast(*args: Tensor) -> List[Tensor]:
#     """
#     This function broadcasts the input arguments to match the shape of each other.
#     """
#     shapes = [arg.shape for arg in args]
#     broadcast_shape = compute_broadcast_shape(*shapes)
#     return (expand_to_shape(arg, broadcast_shape) for arg in args)


# def compute_broadcast_shape(*shapes: Tuple[int]) -> Tuple[int]:
#     reversed_shapes = [shape[::-1] for shape in shapes]
#     broadcast_shape = []
#     for dims in zip_longest(*reversed_shapes, fillvalue=1):
#         greater_than_one_dims = [dim for dim in dims if dim > 1]
#         if len(set(greater_than_one_dims)) > 1:
#             raise ValueError("operands could not be broadcast together with shapes " + ' '.join(map(str, shapes)))
#         broadcast_shape.append(max(dims))
#     return tuple(broadcast_shape[::-1])


# def squeeze_first_dim(inp: Any, len: int = 1) -> Union[Array, MultiArray]:
#     assert isinstance(inp, (sfix, cfix, sint, cint, regint, Array, SubMultiArray, MultiArray)), "Input must be a scale(sfix,cfix,sint,cint,regint) or a array(Array,SubMultiArray,MultiArray)"
#     if isinstance(inp, (sfix, cfix, sint, cint, regint)):
#         res = Array(len, type(inp))
#         res.assign_all(inp)
#     else:
#         shape = (inp.length,) if isinstance(inp, Array) else inp.sizes
#         res = MultiArray([len, *shape], inp.value_type)
#         for i in range(len):
#             res[i] = inp
#     return res


# def expand_to_shape(inp: Tensor, target_shape: Tuple[int]) -> Tensor:
#     """
#     This function expands the inp to match the target_shape using broadcasting rules.
#     """
#     assert isinstance(inp, Tensor), "Input must be a Tensor"
#     input_shape = inp.shape
#     input = inp.value
#     # Calculate the difference in dimensions between the input and target
#     diff_dim = len(target_shape) - len(input_shape)

#     # If the input tensor has fewer dimensions than target shape, add dimensions to the front
#     if diff_dim > 0:
#         for _ in range(diff_dim):
#             input = squeeze_first_dim(input)

#     res = MultiArray(list(target_shape), input.value_type)

#     def expand_dim(obj: Union[Array, MultiArray], res: MultiArray, dim: int) -> Union[Array, MultiArray]:
#         """
#         This is a recursive helper function to expand the list along the specified dimension.
#         """
#         # If the current dimension is less than the number of dimensions in target shape
#         if dim >= len(target_shape):
#             return obj

#         # Get the shape of the current input tensor
#         current_shape = (obj.length,) if isinstance(obj, Array) else obj.sizes
#         # If the size at the current dimension is 1, replicate the element to match target size
#         if current_shape[0] == 1 and target_shape[dim] != 1:
#             obj = squeeze_first_dim(obj[0], target_shape[dim])
#         # Continue to expand each item in the current list if not in the last dimension
#         if dim + 1 < len(target_shape):
#             for i in range(target_shape[dim]):
#                 res[i] = expand_dim(obj[i], res[i], dim + 1)
#             return res
#         else:
#             return obj

    # return Tensor(expand_dim(input, res, 0))
    
      # @buildingblock("softmax-forward")
    # def softmax(self, dim=-1):
    #     @backwardbuildingblock(get_program().globalbuildingblock[:-16]+"-sofxmax-backward")
    #     def propagate(dl_doutputs, operation):
    #         dl_dy, = dl_doutputs
    #         output = tensors[operation.outputs[0]]
    #         if self.req_grad:
    #             if isinstance(self.value, MultiArray):
    #                 inter_mul1, inter_mul2, inter_sum,inter1,inter2= operation.intermediate[-5], operation.intermediate[-4], operation.intermediate[-3],operation.intermediate[-2],operation.intermediate[-1]
    #                 # dl_dx = softmax(x)*(   dl_dy    -    (dl_dy*softmax(x)).sum(dim=-1)  )
    #                 dl_dy.element_wise_mul(output.value, inter_mul1)
    #                 inter_mul1.sum(dim, res=inter_sum, keepdims=True)
    #                 _, v1, v2 = reconst_dims(dl_dy, inter_sum)
                    
    #                 boardcasted_multiarray_sub(v1,v2, inter2,inter1)
    #                 output.value.element_wise_mul(inter1, inter_mul2)
    #                 dl_d[operation.inputs[0]][:] += inter_mul2[:]
    #             else:
    #                 res = output.value[:]*(dl_dy[:]-sum(output.value[:]*dl_dy[:]))
    #                 dl_d[operation.inputs[0]][:] += res
    #     # forward        
    #     global op_id 
    #     if prepare:
    #         if isinstance(self.value, Array):
    #             output = Tensor(Array(self.sizes[0], self.value_type), req_grad=self.req_grad)
    #             inter = []
    #         else:
    #             output = Tensor(MultiArray(self.sizes, self.value_type), req_grad=self.req_grad)
    #             new_sizes = [*self.sizes]
    #             new_sizes[dim], new_sizes[-1] = new_sizes[-1], new_sizes[dim]
    #             per_x, per_res = MultiArray(new_sizes, self.value_type), MultiArray(new_sizes, self.value_type)
    #             inter = [per_x, per_res]
    #         if self.req_grad:
    #             if isinstance(self.value, MultiArray):
    #                 if dim in [-1 , len(self.sizes)-1]:
    #                     new_sizes = self.sizes[:dim] +(1,) 
    #                 else:
    #                     new_sizes = self.sizes[:dim] +(1,) +self.sizes[dim+1:]
    #                 inter_mul1, inter_mul2, inter_sum = MultiArray(self.value.sizes, self.value.value_type), MultiArray(
    #                     self.value.sizes, self.value.value_type), MultiArray(new_sizes, self.value.value_type)
    #                 dims, v1, v2 = reconst_dims(output.value, inter_sum)
    #                 target_size = v1.tuple_permute(output.value.sizes, get_permute(len(output.sizes), dims))
    #                 inter1,inter2=MultiArray(output.sizes,self.value_type),MultiArray(target_size,self.value_type)
    #                 inter += [inter_mul1, inter_mul2, inter_sum,inter1,inter2]
    #             operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=inter)
    #         else:
    #             operation = Operation(inputs=[self.name], outputs=[output.name], propagate=fake_propagate, intermediate=inter)
    #         gradient_operation.append(operation)
    #         operation_id = len(gradient_operation) - 1

    #         op_id_store[op_id] = operation_id
    #         op_id += 1
    #     else:
    #         operation = gradient_operation[op_id_store[op_id]]
    #         inputs = operation.inputs
    #         outputs = operation.outputs
    #         input = tensors[inputs[0]]
    #         output = tensors[outputs[0]]
    #         if isinstance(self.value, Array):
    #             softmax_last_dim(input.value, dim, output.value)
    #         else:
    #             softmax_last_dim(input.value, dim, output.value, [operation.intermediate[0],operation.intermediate[1]])
    #         op_id += 1
    #     # record the input and output of the op
    #     return output
