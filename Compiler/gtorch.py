from Compiler.tensor import *
from Compiler.functional import *
from Compiler.types import *

def is_tensor(obj):
    ''' Returns True if 'obj' is a PyTorch tensor. '''
    return isinstance(obj, Tensor)

def numel(input):
    ''' Returns the total number of elements in the 'input' tensor. '''
    # assert isinstance(input, Tensor)
    return input.numel()

def zeros(*sizes, value_type = sfix, req_grad = False):
    ''' Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument 'size'. '''
    return Tensor.zeros(*sizes, value_type = value_type, req_grad = req_grad)

def zeros_like(input):
    ''' Returns a tensor filled with the scalar value 0, with the same size as 'input'.  '''
    # assert isinstance(input, Tensor)
    return Tensor.zeros(*input.size(), value_type = input.value_type, req_grad = input.req_grad)

def ones(*sizes, value_type = sfix, req_grad = False):
    ''' Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument 'size'. '''
    return Tensor.ones(*sizes, value_type = value_type, req_grad = req_grad)

def ones_like(input):
    ''' Returns a tensor filled with the scalar value 1, with the same size as 'input'.  '''
    # assert isinstance(input, Tensor)
    return Tensor.ones(*input.size(), value_type = input.value_type, req_grad = input.req_grad)

def arange(start, end, step =1, value_type = sfix, req_grad = False):
    ''' Returns a 1-D tensor of size ceil((end-start)/step) with values from the interval '[start, end)' taken with common difference 'step' beginning from start. '''
    return Tensor.arange(start = start, end = end, step = step, value_type = value_type, req_grad = req_grad)

def linspace(start, end, steps, value_type = sfix, req_grad = False):
    ''' Creates a one-dimensional tensor of size 'steps' whose values are evenly spaced from 'start' to 'end', inclusive. '''
    sizes = steps
    step = int((end - start)/(sizes - 1))
    res_value = Array(sizes, value_type)
    @for_range(start, end, step)
    def _(i):
        res_value.assign_vector(value_type(i), (i - start)/step)
    res_value.assign_vector(value_type(end), sizes - 1)
    res = Tensor(res_value, req_grad = req_grad)
    return res

def logspace(start, end, steps, base = 10, value_type = sfix, req_grad = False):
    ''' Creates a one-dimensional tensor of size 'steps' whose values are evenly spaced from base ** start to base ** end, inclusive, on a logarithmic scale with base 'base'. '''
    sizes = steps
    start = base ** start
    end = base ** end
    step = int((end - start)/(sizes - 1))
    res_value = Array(sizes, value_type)
    @for_range(start, end, step)
    def _(i):
        res_value.assign_vector(value_type(i), (i - start)/step)
    res_value.assign_vector(value_type(end), sizes - 1)
    res = Tensor(res_value, req_grad = req_grad)
    return res

def eye(m, n = None, value_type = sfix, req_grad = False):
    ''' Returns a 2-D tensor with ones on the diagonal and zeros elsewhere. '''
    return Tensor.eye(m = m, n = n, value_type = value_type, req_grad = req_grad)

def empty(*sizes, value_type = sfix, req_grad = False): # zeros
    ''' Returns a tensor filled with uninitialized data. The shape of the tensor is defined by the variable argument 'size'. '''
    return Tensor.zeros(*sizes, value_type = value_type, req_grad = req_grad)

def empty_like(input): # zeros_like
    ''' Returns an uninitialized tensor with the same size as 'input'.  '''
    return Tensor.zeros(*input.size(), value_type = input.value_type, req_grad = input.req_grad)

def empty_strided(sizes, stride, value_type = sfix, req_grad = False): # stride是各维度的步长，len(stride) = len(sizes)，第一个元素是0
    ''' Creates a tensor with the specified 'size' and 'stride' and filled with undefined data. '''
    res = Tensor.zeros(sizes, value_type = value_type, req_grad = req_grad)
    for i in range(len(sizes)):
        index = index * sizes[-i] if i > 0 else 1 # 当前维度赋值起始位置（从最低维度开始）
        v = res.value.get_vector(0, index) # 获取该维度下第一个向量
        for j in range(sizes[-i - 1] - 1):
            v = v + stride[-i - 1]
            res.value.assign_vector(v, index + j * len(v)) # 根据当前维度的步长填充该维度
    return res

def full(sizes, fill_value, value_type = sfix, req_grad = False):
    ''' Creates a tensor of size 'size' filled with 'fill_value'. '''
    res = Tensor.zeros(sizes, value_type = value_type, req_grad = req_grad)
    res.value.assign_all(value_type(fill_value))
    return res

def full_like(input, fill_value, value_type = sfix, req_grad = False):
    ''' Returns a tensor with the same size as 'input' filled with 'fill_value'. '''
    res = Tensor.zeros(*input.size(), value_type = value_type, req_grad = req_grad)
    res.value.assign_all(value_type(fill_value))
    return res

def adjoint(input): # input.transpose(-2, -1)
    ''' Returns a view of the tensor conjugated and with the last two dimensions transposed. '''
    return input.transpose(-2, -1)

def cat(*inputs, dim = 0):
    ''' Concatenates the given sequence of 'seq' tensors in the given dimension. '''
    if len(inputs) == 1:
        return inputs[0][0]
    inputs = list(inputs)
    res = inputs[0]
    for i in range(len(inputs) - 1):
        res = res.concat(inputs[i + 1], dim)
    return res

def chunk(input, chunks, dim=0):
    ''' Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor. '''
    return input.chunk(chunks, dim)

def dsplit(input, *indices_or_sections):
    ''' Splits 'input', a tensor with three or more dimensions, into multiple tensors depthwise according to 'indices_or_sections'. '''
    return tensor_split(input, *indices_or_sections, dim = 2)

def column_stack(*tensors):
    ''' Creates a new tensor by horizontally stacking the tensors in tensors. '''
    if len(tensors) == 1:
        res = tensors[0][0]
        return reshape(res, [numel(res), 1])
    tensors = list(tensors)
    if len(tensors[0].size()) == 1:
        res = reshape(tensors[0], [numel(tensors[0]), 1])
    else:
        res = tensors[0]
    for i in range(len(tensors) - 1):
        if len(tensors[i + 1].size()) == 1:
            input = reshape(tensors[i + 1], [numel(tensors[i + 1]), 1])
        else:
            input = tensors[i + 1]
        res = res.concat(input, dim = 1)
    return res

def dstack(*tensors):
    ''' Stack tensors in sequence depthwise (along third axis). '''
    if len(tensors) == 1:
        res = tensors[0]
        if res.dim == 1:
            res = unsqueeze(res, res.dim)
            res = unsqueeze(res, res.dim)
        elif res.dim == 2:
            res = unsqueeze(res, res.dim)
        return res
    else:
        tensors = list(tensors)
        for i in range(len(tensors)):
            if tensors[i].dim == 1:
                tensors[i] = unsqueeze(tensors[i], tensors[i].dim)
                tensors[i] = unsqueeze(tensors[i], tensors[i].dim)
            elif tensors[i].dim == 2:
                tensors[i] = unsqueeze(tensors[i], tensors[i].dim)
        res = cat(*tensors, dim = 2)
        return res

def gather(input, dim, index):
    ''' Gathers values along an axis specified by dim. '''
    return input.gather(dim, index)

def hsplit(input, *indices_or_sections):
    ''' Splits 'input', a tensor with one or more dimensions, into multiple tensors horizontally according to 'indices_or_sections'. '''
    if input.dim == 1:
        return tensor_split(input, *indices_or_sections, dim = 0)
    else:
        return tensor_split(input, *indices_or_sections, dim = 1)

def hstack(*tensors):
    ''' Stack tensors in sequence horizontally (column wise). '''
    if len(tensors) == 1:
        return tensors[0][0]
    tensors = list(tensors)
    res = tensors[0]
    for i in range(len(tensors) - 1):
        if len(tensors[i + 1].size()) == 1:
            res = res.concat(tensors[i + 1], dim = 0)
        else:
            res = res.concat(tensors[i + 1], dim = 1)
    return res

def movedim(input, source, destination):
    ''' Moves the dimension(s) of 'input' at the position(s) in 'source' to the position(s) in 'destination'. '''
    source_dims = source if isinstance(source, tuple) else [source]
    destination_dims = destination if isinstance(destination, tuple) else [destination]
    dims = list(range(input.dim))
    for source, destination in zip(source_dims, destination_dims):
        dims.pop(source)
        dims.insert(destination, source)
    res = permute(input, *dims)
    return res

def narrow(input, dim, start, length):
    ''' Returns a new tensor that is a narrowed version of 'input' tensor. The dimension 'dim' is input from 'start' to 'start + length'. '''
    tmp = tensor_split(input, start, start + length, dim = dim)
    res = tmp[1]
    return res

def permute(input, *dims):
    ''' Returns a view of the original tensor 'input' with its dimensions permuted. '''
    return input.permute(*dims)

def reshape(input, *shape):
    ''' Returns a tensor with the same data and number of elements as 'input', but with the specified shape. '''
    return input.reshape(*shape)

def select(input, dim, index):
    ''' Slices the 'input' tensor along the selected dimension at the given index. '''
    slices = [slice(None)] * input.dim
    slices[dim] = index
    slices = tuple(slices)
    return input[slices]

def squeeze(input, dim = None):
    ''' Returns a tensor with all specified dimensions of 'input' of size 1 removed. '''
    return input.squeeze(dim)

def stack(*tensors, dim = 0):
    ''' Concatenates a sequence of tensors along a new dimension. '''
    if dim == tensors[0].dim:
        tensors = list(tensors)
        for i in range(len(tensors)):
            tensors[i] = unsqueeze(tensors[i], dim)
        res = cat(*tensors, dim = dim)
        return res
    else:
        res = cat(*tensors, dim = dim)
        shape = list(res.shape)
        shape[dim] //= len(tensors)
        shape.insert(dim, len(tensors))
        res = reshape(res, shape)
        return res

def tensor_t(input): # t
    ''' Expects 'input' to be <= 2-D tensor and transposes dimensions 0 and 1. '''
    if input.dim < 2:
        return input
    else:
        return transpose(input, 0, 1)

def take(input, *indices):
    ''' Returns a new tensor with the elements of 'input' at the given indices. '''
    if len(indices) == 1:
        indices = indices[0]
    else:
        indices = list(indices)
    value = Array(len(indices), input.value.value_type)
    for i in range(len(indices)):
        v = input.value.get_vector(indices[i], 1)
        value.assign_vector(v, i)
    res = Tensor(value, req_grad = input.req_grad)
    return res

def tensor_split(input, *indices_or_sections, dim = 0):
    ''' Splits a tensor into multiple sub-tensors, along dimension 'dim' according to the indices or number of sections specified by 'indices_or_sections'. '''
    if len(indices_or_sections) == 1: # indices_or_sections为一个数sections，则将input分为sections块，注意与chunk分块方式不同
        sections = indices_or_sections[0] #分块数量
        former_dim_size = input.shape[dim] // sections + 1 # 前面部分的维度长度
        latter_dim_size = former_dim_size - 1 # 后面部分的维度长度
        former_sections = input.shape[dim] % sections # 前面部分的块数
        latter_sections = sections - former_sections # 后面部分的块数
        former_new_size = input.shape[:dim] + (former_dim_size,) + input.shape[dim+1:] # 前面部分的形状
        latter_new_size = input.shape[:dim] + (latter_dim_size,) + input.shape[dim+1:] # 后面部分的形状
        if input.shape[dim + 1:]:
            stride = reduce(lambda x,y:x*y,input.shape[dim+1:]) # 指定维度后面的元素数量
        else:
            stride = 1
        prefix_total = input.value.total_size() // stride // input.shape[dim] # 高维乘积，外层循环次数

        if len(input.shape) == 1:
            output = [Tensor(Array(former_new_size[0], sfix), req_grad=input.req_grad) for i in range(former_sections)]
            for i in range(latter_sections):
                output.append(Tensor(Array(latter_new_size[0], sfix), req_grad=input.req_grad))
        else:
            output = [Tensor(MultiArray(former_new_size, sfix), req_grad=input.req_grad) for i in range(former_sections)]
            for i in range(latter_sections):
                output.append(Tensor(MultiArray(latter_new_size, sfix), req_grad=input.req_grad))

        @for_range(prefix_total)
        def _(i):
            for j in range(input.shape[dim]):
                v = input.value.get_vector(i*input.shape[dim]*stride+j*stride,stride)
                dim_size = former_dim_size if j//former_dim_size < former_sections else latter_dim_size
                if j//former_dim_size >= former_sections and sections != 1: # 前面部分
                    output[former_sections + (j - former_sections * former_dim_size) // latter_dim_size].value.assign_vector(v,i*dim_size*stride+((j - former_sections * former_dim_size) % dim_size)*stride)
                else: # 后面部分
                    output[j//former_dim_size].value.assign_vector(v,i*dim_size*stride+(j%dim_size)*stride)
        return output
    
    else: # 根据索引列表对input进行切分，索引为切点
        indices = list(indices_or_sections)
        dim_size_list = [indices[0]]
        count = indices[0]
        for i in range(len(indices) - 1):
            if count + indices[i + 1] - indices[i] > input.shape[dim]: # 处理索引超过维度长度的情况
                dim_size_list.append(input.shape[dim] - count)
                dim_size_list.append(indices[i + 1] - indices[i] - (input.shape[dim] - count))
                count += (indices[i + 1] - indices[i])
                break
            else:
                dim_size_list.append(indices[i + 1] - indices[i])
                count += (indices[i + 1] - indices[i])
        if count < input.shape[dim]:
            dim_size_list.append(input.shape[dim] - indices[len(indices) - 1]) # 指定维度下每部分的长度
        new_size_list = [input.shape[:dim] + (dim_size_list[i],) + input.shape[dim + 1:] for i in range(len(dim_size_list))] # 每部分的形状
        if input.shape[dim + 1:]:
            stride = reduce(lambda x,y:x*y,input.shape[dim+1:]) # 指定维度后面的元素数量
        else:
            stride = 1
        prefix_total = input.value.total_size() // stride // input.shape[dim] # 高维乘积，外层循环次数

        output = []
        if len(input.shape) == 1:
            for i in range(len(new_size_list)):
                output.append(Tensor(Array(new_size_list[i][0], sfix), req_grad=input.req_grad))
        else:
            for i in range(len(new_size_list)):
                output.append(Tensor(MultiArray(new_size_list[i], sfix), req_grad=input.req_grad))

        @for_range(prefix_total)
        def _(i):
            index, cnt = 0, 0 # 分别记录块的索引以及之前元素总数
            for j in range(input.shape[dim]):
                if j >= cnt + dim_size_list[index]: # 若符合条件则进入下一块
                    cnt += dim_size_list[index]
                    index += 1
                if index >= input.shape[dim]: # 索引溢出的块用0填充
                    output[index].value.assign_vector(0,i*dim_size*stride+((j - cnt)%dim_size)*stride)
                else:
                    v = input.value.get_vector(i*input.shape[dim]*stride+j*stride,stride)
                    dim_size = dim_size_list[index]
                    output[index].value.assign_vector(v,i*dim_size*stride+((j - cnt)%dim_size)*stride)
        return output

def tile(input, *dims):
    ''' Constructs a tensor by repeating the elements of 'input'. The 'dims' argument specifies the number of repetitions in each dimension. '''
    if len(dims) == 1:
        sizes = [input.shape[0] * dims[0]]
        res = Tensor.zeros(sizes, value_type = input.value_type, req_grad = input.req_grad)
        v = input.value.get_vector()
        for i in range(dims[0]):
            res.value.assign_vector(v, i * input.shape[0])
        return res
    else:
        dims = list(dims)
        sizes = [input.shape[i] * dims[i] for i in range(input.dim)]
        res = Tensor.zeros(sizes, value_type = input.value_type, req_grad = input.req_grad)
        indices = [0] * input.dim # 索引列表用于迭代
        for i in range(numel(res)):
            original_indices = tuple(index % dim for index, dim in zip(indices, input.shape)) # input中的索引
            res[tuple(indices)] = input[original_indices] # 赋值
            for j in range(input.dim - 1, -1, -1): # 更新索引列表
                indices[j] += 1
                if indices[j] < res.shape[j]:
                    break
                indices[j] = 0
        return res

def transpose(input, dim0, dim1):
    ''' Returns a tensor that is a transposed version of 'input'. The given dimensions 'dim0' and 'dim1' are swapped. '''
    return input.transpose(dim0, dim1)

def unbind(input, dim = 0):
    ''' Removes a tensor dimension. '''
    sections = input.shape[dim]
    return tensor_split(input, sections, dim = dim)

def unsqueeze(input, dim):
    ''' Returns a new tensor with a dimension of size one inserted at the specified position. '''
    return input.unsqueeze(dim)

def vsplit(input, *indices_or_sections):
    ''' Splits 'input', a tensor with two or more dimensions, into multiple tensors vertically according to 'indices_or_sections'. Each split is a view of 'input'. '''
    return tensor_split(input, *indices_or_sections)

def vstack(*tensors):
    ''' Stack tensors in sequence vertically (row wise). '''
    if len(tensors) == 1:
        return tensors[0][0]
    tensors = list(tensors)
    for i in range(len(tensors)):
        if len(tensors[i].shape) == 1:
            tensors[i] = unsqueeze(tensors[i], 0)
    res = cat(*tensors)
    return res
