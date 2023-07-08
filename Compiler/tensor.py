from glob import glob
import math
import re
from turtle import forward, shape

from Compiler import mpc_math, util
from Compiler.types import *
from Compiler.types import _unreduced_squant
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
from Compiler.GC.types import sbitint
from functools import reduce
from typing import List, NamedTuple, Callable, Dict, Optional


_name = 1

#sotre of tensors involved in computation process
tensors  =  {}
#store of operators invovled in computation process, these operators are topologically sotred
gradient_operation : List[Operation] = []
#sotre of gradients involved in computation process, their types are tensors without gradient
dl_d =  {}
#the flag indicates whether initialize gradients for tensors
is_train = True
#the flag indicated that we are in prepration phase, i.e. initialize inputs and outputs for each operators
prepare = True
#op_id is used for extracting the references of inputs and outputs of one opertator
op_id = 0
#op_id_store stores the correlation among op_ids and operation ids.
op_id_store = {}


def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name

def train():
    global prepare
    prepare = False

def same_shape(sizes1, sizes2):
    if len(sizes1) != len(sizes2):
        return False
    for i in range(0, len(sizes1)):
        if sizes1[i] != sizes2[i]:
            return False
    return True

def element_wise_mul(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself =  dl_d[inputs[0]]# partial derivate of r = self*other
        dl_dother = dl_d[inputs[1]] # partial derivate of r = self*other
        dl_dself[:] += dl_dx[:] * other.value[:] #todo
        dl_dother[:] += dl_dx[:] * self.value[:] # todo
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    if prepare:    
        new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
        output = Tensor(new_value)
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        global op_id
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        output.value[:] = input1.value[:] * input2.value[:] #todo        
        op_id += 1

    # record the input and output of the op
    return output

def mat_mul(self, other):

    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself =  dl_d[inputs[0]]# partial derivate of r = self*other
        dl_dother = dl_d[inputs[1]] # partial derivate of r = self*other
        dl_dself[:] += dl_dx[:] * other.value[:] #todo
        dl_dother[:] += dl_dx[:] * self.value[:] # todo
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    if prepare:    
        new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
        output = Tensor(new_value)
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        global op_id
        op_id_store[op_id] = operation_id
        op_id += 1
    else:
        operation = gradient_operation[op_id_store[op_id]]
        inputs = operation.inputs
        outputs = operation.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        output.value[:] = input1.value[:] * input2.value[:] #todo        
        op_id += 1

    # record the input and output of the op
    return output    

def ops_add(self, other):
    x = Tensor(self.value + other.value)
    print(f'{x.name} = {self.name} + {other.name}')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Tensor(1.)
        dx_dother = Tensor(1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]

    # record the input and output of the op
    operation = Operation(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_operation.append(operation)
    return x

def ops_sub(self, other):
    x = Tensor(self.value - other.value)

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        dx_dself = Tensor(1.)
        dx_dother = Tensor(-1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]

    # record the input and output of the op
    operation = Operation(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_operation.append(operation)
    return x



class Tensor():
    def __init__(self, value, name=None, req_grad = False, is_grad = False):
        assert isinstance(value, Array) or isinstance(value, MultiArray)
        self.value = value
        self.name = name or fresh_name()
        self.shape = value.sizes
        self.req_grad = req_grad
        if is_train and not is_grad:
            self.grad = self.value.same_shape()
            self.grad.assign_all(0)
            dl_d[self.name] = self.grad
            tensors[self.name] = self

    def __repr__(self):
        return self.value
    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value, name=None):
        var = Tensor(value, name)
        return var

    def backward(self):
        length = len(gradient_operation)
        index = 0
        dl_d[self.name].assign_all(1)
        # the following loop only runs once in the training process due the semantice of @for_range
        def gather_grad(entries):
            return [dl_d[entry]  for entry in entries]        
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
        if not same_shape(self.size(), other.size()):
            exit(0)
        return element_wise_mul(self, other)

    def mul(self, other):
        # todo
        return self
    
    def mv(self, other):
        # todo
        return self
    
    def mm(self, other):
        #todo
        return self

    def dot(self, other):
        #todo
        return self
    
    def matmul(self, other):
        # todo, may not implement
        return self
    
    def div(self, other):
        #todo
        return self

    def __add__(self, other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)

    def __getitem__(self, index):
        #todo
        return self    
    
    def view(self):
        #todo
        return self
    
    def squeeze(self):
        #todo
        return self

    def unsqueeze(self):
        #todo
        return self

    def gather(self):
        #todo
        return self
    
    def reshape(self, sizes):
        #todo
        return self

    def permute(self, sizes):
        #todo 
        return self
    
    def transpose(self):
        #todo
        return self

    def concate(self, other):
        #todo
        return self

    def abs(self):
        #todo
        return self

    def exp(self):
        #todo
        return self

    def log(self):
        #todo
        return self

    def pow(self, pow):
        #todo
        return self

    def cos(self):
        #todo
        return self

    def sin(self):
        #todo
        return self

    def mean(self):
        #todo
        return self

    def sum(self):
        #todo
        return self

    def std(self):
        #todo
        return self

    def var(self):
        #todo
        return self    

    
    def size(self):
        return self.value.sizes

    def zero_grad(self):
        self.grad.assign_all(0)

    
# reset operation
def reset_operation():
    gradient_operation.clear()

#call this function after each iteration
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


class Operation(NamedTuple):
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : 'Callable[List[Tensor], List[Tensor]]'
    # forward : 'Callable[List[Tensor], List[Tensor]]'

