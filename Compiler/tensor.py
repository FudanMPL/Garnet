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
from Compiler.GC.types import sbitint
from functools import reduce
from typing import List, NamedTuple, Callable, Dict, Optional, Union, Tuple, Any

_name = 1
class Operation(NamedTuple):
    inputs : List[str]
    outputs : List[str]
    
    # apply chain rule
    propagate : 'Callable[List[Tensor], List[Tensor]]'
    # forward : 'Callable[List[Tensor], List[Tensor]]'
    intermediate : List = [] 
     
#sotre of tensors involved in computation process
tensors  =  {}
#store of operators invovled in computation process, these operators are topologically sotred
gradient_operation : List[Operation] = []
#sotre of gradients involved in computation process, their types are tensors without gradient
dl_d  =  {}
#the flag indicates whether initialize gradients for tensors
is_train = True
#the flag indicated that we are in prepration phase, i.e. initialize inputs and outputs for each operators
prepare = True
#op_id is used for extracting the references of inputs and outputs of one opertator
op_id = 0
#op_id_store stores the correlation among op_ids and operation ids.
op_id_store = {}

# def matrix_reconst(self, mat):
#     r = mat.value.length/mat.size[0]
#     c = mat.size[0]
#     new_matrix = MultiArray([r, c], mat.value.value_type)
#     @for_range(r)
#     def _(i):
#         @for_range(c)
#         def _(j):
#             new_matrix[i][j] = 
#     return new_matrix
    

def element_wise_add(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself =  dl_d[inputs[0]] # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]] # partial derivate of r = 1
        dl_dself[:] += dl_dx[:]
        dl_dother[:] += dl_dx[:]
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:    
        new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
        output = Tensor(new_value)
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
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
        
        len1 = input1.value.total_size()
        len2 = input2.value.total_size()
        v1 = input1.value.get_vector(0, len1)
        v2 = input2.value.get_vector(0, len2)
        if len1 < len2:
            len1, len2 = len2, len1
            v1, v2 = v2, v1

        # print(type(v1))
        # for i in range(0, int(len1/len2)):
        #     for j in range(0, len2):
        #         v1[i+j] += v2[j]

        output.value.assign_vector(v1)
        op_id += 1# record the input and output of the op
    return output

def element_wise_sub(self, other):
    # backward
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs, operation):
        dl_dx, = dl_doutputs
        inputs = operation.inputs
        dl_dself =  dl_d[inputs[0]] # partial derivate of r = 1
        dl_dother = dl_d[inputs[1]] # partial derivate of r = -1
        dl_dself[:] += dl_dx[:]
        dl_dother[:] += - dl_dx[:]
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # forward
    global op_id
    if prepare:    
        new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
        output = Tensor(new_value)
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
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
        output.value[:] = input1.value[:] - input2.value[:] #todo        
        op_id += 1# record the input and output of the op
    return output

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
    global op_id
    if prepare:    
        new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
        output = Tensor(new_value)
        operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
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
        output.value[:] = input1.value[:] * input2.value[:] #todo        
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
            output = Tensor(new_value)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value)
            
        operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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

def ops_sin(self):
    @buildingblock(get_program().globalbuildingblock)
    def propagate(dl_doutputs,oparation):
        dl_dx, = dl_doutputs
        dx_dself = Tensor(mpc_math.scos(self.value))
        dl_dself = dl_dx * dx_dself
        return [dl_dself]

    if prepare:
        new_value=MultiArray([self.value.sizes[0], self.value.sizes[1]],self.value.value_type)
        output = Tensor(new_value)
        operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
        gradient_operation.append(operation)
        operation_id = len(gradient_operation) - 1
        global op_id
        op_id_store[op_id] = operation_id
        op_id+=1
    else:
        operation=gradient_operation[op_id_store[op_id]]
        inputs=operation.inputs
        outputs=operation.outputs
        input=tensors[inputs[0]]
        output=tensors[outputs[0]]
        output.value[:]=Tensor(mpc_math.ssin(self.value))
        op_id+=1
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
            output = Tensor(new_value)
        else:
            new_value = MultiArray(self.value.sizes, self.value.value_type)
            output = Tensor(new_value)
            
        operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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

# def compare_shape(shape1, shape2):
#     if len(shape1) == len(shape2):

#     else:
#         return -1

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



    def set_req_grad(self, req_grad):
        self.req_grad = req_grad

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
        if isinstance(other, (int, float)):
            return ops_mul_constant(self, other)
        return element_wise_mul(self, other)

    def mul(self, other):
        # todo
        return self
    
    def mv(self, other):
        # todo
        return self
    
    def mm(self, other):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            dl_dself =  dl_d[inputs[0]] # C=AB partial derivate of dA=dC*B^T
            dl_dother = dl_d[inputs[1]] # C=AB partial derivate of dB=A^T*dC
            dl_dself[:] += dl_dx[:]
            dl_dother[:] += dl_dx[:]
            dl_dinputs = [dl_dself, dl_dother]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            assert len(self.shape)==len(other.shape)==2 and self.shape[1]==other.shape[0],"Invalid Dimension"
            new_value = MultiArray([self.value.sizes[0], other.value.sizes[1]], other.value.value_type)
            output = Tensor(new_value)
            operation = Operation(inputs=[self.name, other.name], outputs=[output.name], propagate=propagate)
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
            @for_range(input1.shape[0])
            def _(i):
                @for_range(input2.shape[1])
                def _(j):
                    tmp=0
                    for_range(input1.shape[1])
                    def _(k):
                        tmp+=(input1.value[i][k]*input2.value[k][j])
                    output.value[i][j]=tmp                  
            op_id += 1# record the input and output of the op
        return output

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
        #todo
        return self

    def __getitem__(self, index):
        #todo
        return Tensor(self.value[index])
    
    def view(self,sizes): 
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare: 
            product=reduce(lambda x,y:x*y,self.shape)
            if isinstance(sizes,int):
                assert sizes==product,"Invalid Dimension"
                new_value=Array(sizes,self.value.value_type)
            else:
                assert all(isinstance(x,int) and x>0 for x in sizes),"Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1)==1,"-1 Occurs More than Once "
                    tmp=reduce(lambda x,y:x*y,sizes)
                    assert product%(-tmp)==0,"Invalid Dimension"
                    sizes[sizes.index(-1)]=product/(-tmp)
                new_value=MultiArray(sizes,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            outputs=operation.outputs
            output=tensors[outputs[0]] 
            output.value.assign(self.value)
            op_id+=1
        return output
            
    
    def squeeze(self,dim=None):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare: 
            if dim:
                new_sizes=list(self.shape)
                assert dim<len(self.shape),"Invalid Dimension"
                del new_sizes[dim]
            else:
                new_sizes=[x for x in self.shape if x!=1]  
            if len(new_sizes)>1:
                new_value=MultiArray(new_sizes,self.value.value_type)
            else:
                assert len(new_sizes)==1 and new_sizes[0]>0,"Invalid Dimension"  
                new_value=Array(new_sizes[0],value_type=self.value.value_type) 
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            outputs=operation.outputs
            output=tensors[outputs[0]]
            output.value.assign(self.value)
            op_id+=1
        return output

    def unsqueeze(self,dim):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_d[operation.inputs[0]].assign(dl_doutputs[0])
        global op_id
        if prepare: 
            new_sizes=list(self.shape)
            assert isinstance(dim,int) and dim<len(self.shape) and dim>=-len(self.shape),"Invalid Dimension"
            new_sizes.insert(dim,1)              
            new_value=MultiArray(new_sizes,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            outputs=operation.outputs
            output=tensors[outputs[0]]
            output.value.assign(self.value)
            op_id+=1
        return output

    def gather(self):
        #todo
        return self
    
    def reshape(self, sizes):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            dl_d[operation.inputs[0]].assign(dl_dy)
        global op_id
        if prepare: 
            product=reduce(lambda x,y:x*y,self.shape)
            if isinstance(sizes,int):
                assert sizes==product,"Invalid Dimension"
                new_value=Array(sizes,self.value.value_type)
            else:
                assert all(isinstance(x,int) and x>0 for x in sizes),"Invalid Dimensiopn"
                if -1 in sizes:
                    assert sizes.count(-1)==1,"-1 Occurs More than Once "
                    tmp=reduce(lambda x,y:x*y,sizes)
                    assert product%(-tmp)==0,"Invalid Dimension"
                    sizes[sizes.index(-1)]=product/(-tmp)
                new_value=MultiArray(sizes,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            outputs=operation.outputs
            output=tensors[outputs[0]] 
            output.value.assign(self.value)
            op_id+=1
        return output


    def permute(self, new_perm): #todo :这里的参数不应该是list类型的new-perm，而应该是*newperm :pytorch中：x.permute(2, 0, 1)
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            dl_dy,=dl_doutputs
            L=len(self.shape)
            inv_new_perm=[None]*L
            for i in range(L):
                inv_new_perm[new_perm[i]]=i #s2[s1[i]]=i
            self.value.permute_without_malloc(dl_d[operation.inputs[0]],inv_new_perm)
        global op_id
        if prepare: 
            assert isinstance(self.value,MultiArray),"Error,Permute operation must be MultiArray"#置换维度，那么肯定是MultiArray吧
            target_size=self.value.tuple_permute(self.shape,new_perm) #just for calling of tuple_permute function
            new_value = MultiArray(target_size,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            outputs=operation.outputs
            output=tensors[outputs[0]]
            self.value.permute_without_malloc(output.value,new_perm) #output的值在参数中传入后被修改
            op_id+=1
        return output
    
    
    def transpose(self):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            if isinstance(dl_doutputs[0],Array):
                dl_d[operation.inputs[0]][:]+=dl_doutputs[0][:]
            else:
                dl_d[operation.inputs[0]][:]+=dl_doutputs[0].transpose()[:]
        global op_id
        if prepare:
            if isinstance(self.value,Array):
                new_value=Array(self.shape[0],self.value.value_type)
            else:
                assert len(self.value.sizes)==2,'Invalid dimension'
                new_sizes=[self.value.sizes[1],self.value.sizes[0]]
                new_value=MultiArray(new_sizes,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            input=tensors[operation.inputs[0]]
            output=tensors[operation.outputs[0]]
            if len(self.shape)==1:#in this case:Array
                output.value[:]=input.value[:]
            else:
                output.value=input.value.transpose()
            op_id+=1
        return output


    def concate(self, other,axis=0):#按照axis指定维度进行拼接
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation):
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            size_pre=reduce(lambda x,y:x*y,input1.shape[axis:])
            size_next=reduce(lambda x,y:x*y,input2.shape[axis:])       
            for i in range(input1.value.length//size_pre):
                input1.grad.assign_vector(dl_doutputs[0].get_vector(i*size_pre,size_pre),i*size_pre)
                input2.grad.assign_vector(dl_doutputs[0].get_vector(i*size_next,size_next),i*size_next)   
        global op_id
        if prepare: 
            assert self.value.value_type is other.value.value_type,"Invalid value_type"
            if isinstance(self.value,Array) and isinstance(other.value,Array):
                target_len=self.value.length + other.value.length
                new_value=Array(target_len,self.value.value_type)
            else:
                assert len(self.shape)==len(other.shape) ,"Inequal Dimension"
                for i in range(len(self.shape)):
                    if i != axis and self.shape[i] != other.shape[i]:
                        raise ValueError("Invalid Dimension") 
                target_size=other.value.shape
                target_size[axis]+=self.value.shape[axis]
                new_value=MultiArray(target_size,self.value.value_type)
            output=Tensor(new_value)
            operation=Operation(inputs=[self.name,other.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1             
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            size_pre=reduce(lambda x,y:x*y,self.shape[axis:])
            size_next=reduce(lambda x,y:x*y,other.shape[axis:])
            input1=tensors[operation.inputs[0]]
            input2=tensors[operation.inputs[1]]
            output=tensors[operation.outputs[0]]
            index=0    
            for i in range(self.value.length//size_pre):
                output.value.assign_vector(input1.value.get_vector(i*size_pre,size_pre),index)
                index+=size_pre
                output.value.assign_vector(input2.value.get_vector(i*size_next,size_next),index)
                index+=size_next
            op_id+=1
        return output

    def abs(self):
        # backward
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs, operation):
            dl_dx, = dl_doutputs
            inputs = operation.inputs
            inter = operation.intermediate[0] # reuse the intervalue in mem
            dl_dself =  dl_d[inputs[0]]
            dl_dself[:] +=  (2 * inter[:] - 1) * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:
            if isinstance(self.value, Array):    
                new_value = Array(self.value.length, self.value.value_type)
                output = Tensor(new_value)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                output = Tensor(new_value)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
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
            operation.intermediate[0].assign_vector(c) # write to mem
            
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
            inter = operation.intermediate[0] # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]
            dl_dself[:] += inter[:] * dl_dx[:]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:    
            if isinstance(self.value, Array):    
                new_value = Array(self.value.length, self.value.value_type)
                output = Tensor(new_value)
                inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                output = Tensor(new_value)
                inter = MultiArray(self.value.sizes, self.value.value_type)
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
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

    def log(self, base = math.e):
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
                output = Tensor(new_value)
                # inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                output = Tensor(new_value)
                # inter = MultiArray(self.value.sizes, self.value.value_type)
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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
                output = Tensor(new_value)
                # inter = Array(self.value.length, self.value.value_type)
            else:
                new_value = MultiArray(self.value.sizes, self.value.value_type)
                output = Tensor(new_value)
                # inter = MultiArray(self.value.sizes, self.value.value_type)
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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
        def propagate(dl_doutputs,operation): #dl_outputs is Tensor.value
            dl_dx,=dl_doutputs
            dl_dself=dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*(-mpc_math.sin(self.value[:]))
        global op_id
        if prepare:
            if isinstance(self.value,Array): #Array is instance of tensor?
                new_value=Array(self.value.length,self.value.value_type) #
                output=Tensor(new_value)
            else:
                new_value=MultiArray(self.value.sizes,self.value.value_type)
                output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            inputs=operation.inputs
            outputs=operation.outputs
            input = tensors[inputs[0]] #input is Tensor
            output = tensors[outputs[0]]
            output.value[:]=mpc_math.cos(input.value[:])
            op_id+=1
            return output

    def sin(self):
        @buildingblock(get_program().globalbuildingblock)
        def propagate(dl_doutputs,operation): #dl_outputs is Tensor.value
            dl_dx,=dl_doutputs
            dl_dself=dl_d[operation.inputs[0]]
            dl_dself[:] += dl_dx[:]*mpc_math.cos(self.value[:])
        global op_id
        if prepare:
            if isinstance(self.value,Array): #Array is instance of tensor?
                new_value=Array(self.value.length,self.value.value_type) #
                output=Tensor(new_value)
            else:
                new_value=MultiArray(self.value.sizes,self.value.value_type)
                output=Tensor(new_value)
            operation=Operation(inputs=[self.name],outputs=[output.name],propagate=propagate)
            gradient_operation.append(operation)
            operation_id=len(gradient_operation)-1
            op_id_store[op_id]=operation_id
            op_id+=1
        else:
            operation=gradient_operation[op_id_store[op_id]]
            inputs=operation.inputs
            outputs=operation.outputs
            input = tensors[inputs[0]] #input is Tensor
            output = tensors[outputs[0]]
            output.value[:]=mpc_math.sin(input.value[:])
            op_id+=1
            return output
    def mean(self):
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
            new_value = Array(1, self.value.value_type)
            output = Tensor(new_value)
            
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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

            output.value[:] = sum(input.value[:]) / self.value.total_size()
            
            op_id += 1
        # record the input and output of the op
        return output

    def sum(self):
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
            new_value = Array(1, self.value.value_type)
            output = Tensor(new_value)
            
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate)
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
            
            output.value[:] = sum(input.value[:])
            
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
            output = Tensor(new_value)
            
            if isinstance(self.value, Array):    
                inter1 = Array(self.value.length, self.value.value_type)
            else:
                inter1 = MultiArray(self.value.sizes, self.value.value_type)
            inter2 = Array(1, self.value.value_type)
            
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter1, inter2])
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
            dmean = operation.intermediate[0] # reuse the intervalue in mem
            dl_dself = dl_d[inputs[0]]
            
            dl_dself[:] += 2 / (self.value.total_size()-1) * dmean[:] * dl_dx[0]
            dl_dinputs = [dl_dself]
            return dl_dinputs
        # forward
        global op_id
        if prepare:    
            new_value = Array(1, self.value.value_type)
            output = Tensor(new_value)
            if isinstance(self.value, Array):    
                inter = Array(self.value.length, self.value.value_type)
            else:
                inter = MultiArray(self.value.sizes, self.value.value_type)
            operation = Operation(inputs=[self.name], outputs=[output.name], propagate=propagate, intermediate=[inter])
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

def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name

def train():
    global prepare
    prepare = False

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

def squeeze_first_dim(inp: Any, len: int=1)-> Union[Array, MultiArray]:
    assert isinstance(inp, (sfix,cfix,sint,cint,regint,Array,SubMultiArray,MultiArray)), "Input must be a scale(sfix,cfix,sint,cint,regint) or a array(Array,SubMultiArray,MultiArray)"
    if isinstance(inp, (sfix,cfix,sint,cint,regint)):
        res = Array(len, type(inp))
        res.assign_all(inp)
    else:
        shape = (inp.length,) if isinstance(inp, Array) else inp.sizes
        res = MultiArray([len,*shape], inp.value_type)
        for i in range(len):
            res[i] = inp
    return res

def expand_to_shape(inp: Tensor, target_shape: Tuple[int])-> Tensor:
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
