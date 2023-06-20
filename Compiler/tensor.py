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
gradient_tape : List[Tape] = []
#sotre of gradients involved in computation process, their types are tensors without gradient
dl_d =  {}
#the flag indicates whether initialize gradients for tensors
is_train = True
#the flag indicated that we are in prepration phase, i.e. initialize inputs and outputs for each operators
prepare = True
#op_id is used for extracting the references of inputs and outputs of one opertator
op_id = 0
#op_id_store stores the correlation among op_ids and tape ids.
op_id_store = {}


def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name

def ops_mul(self, other):
    # forward
    # backward
    def propagate(dl_doutputs, tape):
        dl_dx, = dl_doutputs
        inputs = tape.inputs
        outputs = tape.outputs
        dx_dself =  dl_d[inputs[0]]# partial derivate of r = self*other
        dx_dother = dl_d[inputs[1]] # partial derivate of r = self*other
        dl_dself = dl_dx * dx_dself #todo
        dl_dother = dl_dx * dx_dother # todo
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    if prepare:    
        new_value = MultiArray(self.value.sizes[0], other.value.sizes[1])
        output = Tensor(new_value)
        tape = Tape(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
        gradient_tape.append(tape)
        tape_id = len(gradient_tape) - 1
        global op_id
        op_id_store[op_id] = tape_id
        op_id += 1
    else:
        tape = gradient_tape[op_id_store[op_id]]
        inputs = tape.inputs
        outputs = tape.outputs
        input1 = tensors[inputs[0]]
        input2 = tensors[inputs[1]]
        output = tensors[outputs[0]]
        output.vaule = input1.value * input2.value #todo
        if len(input1.shape) == len(input2.shape):
            x = 1
        elif len(input1.shape) == len(input2.shape)+1:
            x = 1
        elif len(input1.shape)+1 == len(input2.shape):
            x = 1
        
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
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
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
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

#call this function after each iteration
def reset_op_id():
    global op_id
    op_id = 0

class Tensor():
    def __init__(self, value, name=None, is_grad = False):
        assert isinstance(value, Array) or isinstance(value, MultiArray)
        self.value = value
        self.name = name or fresh_name()
        self.shape = value.sizes
        if is_train and not is_grad:
            self.grad = Tensor(self.value, is_grad=True)
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

    def backward():
        len = len(gradient_tape)
        index = 0
        # the following loop only runs once in the training process due the semantice of @for_range
        def gather_grad(entries):
            return [dl_d[entry]  for entry in entries]        
        for i in range(0, len):
            if self.name in gradient_tape[len-i-1].outputs:
                   index = len - i
        for i in range(0, index):
            entry = gradient_tape[index-i-1]
            dl_doutputs = gather_grad(entry.outputs)
            dl_dinputs = entry.propagate(dl_doutputs, entry)
            for input, dl_dinput in zip(entry.inputs, dl_dinputs):
                    dl_d[input] += dl_dinput
        return 0

    # Multiplication of a Variable, tracking gradients
    def __mul__(self, other):
        return ops_mul(self, other)

    def __add__(self, other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)


# reset tape
def reset_tape():
    gradient_tape.clear()

class Tape(NamedTuple):
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : 'Callable[List[Tensor], List[Tensor]]'
    # forward : 'Callable[List[Tensor], List[Tensor]]'

