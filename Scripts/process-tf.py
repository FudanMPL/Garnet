#!/usr/bin/env python3

import sys
from functools import reduce
import operator
import math

filename = sys.argv[1]

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import numpy

graph_def = graph_pb2.GraphDef()
graph_def.ParseFromString(open(filename, mode='rb').read())
tf.import_graph_def(graph_def)
graph = tf.compat.v1.get_default_graph()

first = True
paddings = {}

print('ml.Layer.input_bias = False')

def output(op, layer, prev_input=True):
    global first
    print('named["%s"] = %s' % (op.name, layer))
    print('layers.append(named["%s"])' % op.name)
    if prev_input and not first:
        print('named["%s"].inputs = [named["%s"]]' % (op.name,
                                                      op.inputs[0].name[:-2]))
    first = False

def link(dest, source):
    print('named["%s"] = named["%s"]' % (dest.name, source.name))

def source(dest):
    print('named["%s"] = None' % dest.name)

def activate_bias(op):
    print('named["%s"].input_bias = True' % op.name)

def get_shape(shape):
    res = []
    for x in shape:
        try:
            res.append(int(x))
        except:
            res.append(1)
    return res

def get_valid_padding(input_shape, window, strides):
    return [int(math.ceil((x - y + 1) / z))
            for x, y, z in zip(input_shape, window, strides)]

for op in graph.get_operations():
    if op.inputs:
        shape = get_shape(op.inputs[0].shape)
    else:
        shape = None
    t = op.type
    if t in ('VariableV2', 'Const', 'Assign', 'NoOp', 'Fill', 'VarHandleOp'):
        source(op)
    elif t in ('Reshape', 'Squeeze', 'Identity', 'VarIsInitializedOp', 'ReadVariableOp',
               'AssignVariableOp'):
        link(op, op.inputs[0].op)
    elif t == 'Placeholder':
        source(op)
    elif t == 'MatMul':
        #print (op.inputs[0].shape)
        assert reduce(operator.mul, shape) == op.inputs[1].shape[0]
        output(op, 'ml.Dense(1, %d, %d)' % (op.inputs[1].shape[0],
                                         op.inputs[1].shape[1]))
        shape = [1, int(op.inputs[1].shape[1])]
    elif t == 'Conv2D':
        strides = op.get_attr('strides')
        assert len(strides) == 4
        assert strides[0] == 1
        assert strides[3] == 1
        strides = tuple(strides[1:3])
        input_shape = get_shape(op.inputs[0].shape)
        assert len(input_shape) == 4
        window = [int(x) for x in op.inputs[1].shape]
        padding = op.get_attr('padding').decode('u8')
        if padding not in ('SAME', 'VALID'):
            padding = get_shape(padding)
        if op.inputs[0].op.name in paddings:
            assert padding == 'VALID'
            input_shape = get_shape(op.inputs[0].op.inputs[0].shape)
            p = paddings.pop(op.inputs[0].op.name)
            for i in 0, 6:
                assert p[i] == 0
            padding = [p[2], p[4]]
        output_shape = get_shape(op.outputs[0].shape)
        assert len(output_shape) == 4
        output(op, 'ml.FixConv2d(%s, %s, %s, %s, %s, %s, True, '
               'inputs=[named["%s"]])' % \
               (input_shape, tuple(window), (window[3],), output_shape, strides,
                repr(padding), op.inputs[0].op.name))
    elif t in ('Add', 'AddV2') and op.inputs[1].op.type != 'VariableV2':
        output(op, 'ml.Add([%s])' % ','.join('named["%s"]' % x.op.name
                                             for x in op.inputs), False)
    elif t in ('Add', 'BiasAdd'):
        assert op.inputs[0].op.type in ('MatMul', 'Conv2D')
        activate_bias(op.inputs[0].op)
        link(op, op.inputs[0].op)
    elif t == 'Relu':
        assert len(op.inputs) == 1
        output(op, 'ml.Relu(%s, inputs=[named["%s"]])' % (shape,
                                                          op.inputs[0].op.name))
    elif t == 'Square':
        output(op, 'ml.Square(%s)' % (shape,))
    elif t == 'MaxPool':
        strides = op.get_attr('strides')
        ksize = op.get_attr('ksize')
        padding = str(op.get_attr('padding').decode('u8'))
        output(op, 'ml.MaxPool(%s, %s, %s, "%s")' % (shape, strides, ksize,
                                                     padding))
    elif t == 'AvgPool':
        filter_size = op.get_attr('ksize')
        assert len(filter_size) == 4
        assert filter_size[0] == 1
        assert filter_size[-1] == 1
        input_shape = get_shape(op.inputs[0].shape)
        strides = get_shape(op.get_attr('strides'))
        assert strides[0] == 1
        assert strides[3] == 1
        padding = op.get_attr('padding').decode('u8')
        if padding == 'VALID':
            output_shape = get_valid_padding(input_shape, filter_size, strides)
        elif padding == 'SAME':
            output_shape = [int(math.ceil(x / y))
                            for x, y in zip(input_shape, filter_size)]
        else:
            raise Exception('unknown padding type: %s' % padding)
        output(op, 'ml.FixAveragePool2d(%s, %s, %s, %s)' %
               (input_shape, output_shape, filter_size[1:3], strides[1:3]))
    elif t == 'ArgMax':
        assert len(op.inputs) == 2
        shape = get_shape(op.inputs[0].shape)
        dim = int(op.inputs[1].op.get_attr('value').int_val[0])
        for i in range(1, len(shape)):
            if i != dim:
                assert shape[i] == 1
        output(op, 'ml.Argmax((1, %s))' % shape[dim])
    elif t == 'ConcatV2':
        assert len(op.inputs) == 3
        dim = int(op.inputs[2].op.get_attr('value').int_val[0])
        output(op, 'ml.Concat([%s], %s)' % (
            ','.join('named["%s"]' % x.name[:-2] for x in op.inputs[:2]), dim),
               prev_input=False)
    elif t in ('FusedBatchNorm', 'FusedBatchNormV3'):
        output(op, 'ml.FusedBatchNorm(%s, inputs=[named["%s"]])' %
               (get_shape(op.inputs[0].shape), op.inputs[0].op.name))
    elif t == 'Pad':
        paddings[op.name] = numpy.fromstring(op.inputs[1].op.get_attr('value').
                                            tensor_content, 'int32').tolist()
        link(op, op.inputs[0].op)
    else:
        raise Exception('unknown type: %s' % t)

if paddings:
    raise Exception('padding layers only supported before valid convolution:',
                    paddings)
