"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: tpu_partition_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar

def tpu_partitioned_input(inputs, partition_dim=0, name=None):
  r"""An op that groups a list of partitioned inputs together. This op

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      A list of partitioned inputs which must have the same shape.
    partition_dim: An optional `int`. Defaults to `0`.
      An integer describles which dimension is partitioned. -1 means
      those inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedInput", name, inputs, "partition_dim",
        partition_dim)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_input_eager_fallback(
          inputs, partition_dim=partition_dim, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedInput", inputs=inputs, partition_dim=partition_dim,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"), "T", _op._get_attr_type("T"),
              "partition_dim", _op._get_attr_int("partition_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUPartitionedInput = tf_export("raw_ops.TPUPartitionedInput")(_ops.to_raw_op(tpu_partitioned_input))


def tpu_partitioned_input_eager_fallback(inputs, partition_dim, name, ctx):
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_partitioned_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), ctx, [])
  _inputs_flat = list(inputs)
  _attrs = ("N", _attr_N, "T", _attr_T, "partition_dim", partition_dim)
  _result = _execute.execute(b"TPUPartitionedInput", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedInput", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def tpu_partitioned_output(inputs, num_splits, partition_dim=0, name=None):
  r"""An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned

  outputs outside the XLA computation.

  Args:
    inputs: A `Tensor`.
      A tensor which represents the full shape of partitioned tensors.
    num_splits: An `int` that is `>= 1`.
    partition_dim: An optional `int`. Defaults to `0`.
      An integer describles which dimension is partitioned.
    name: A name for the operation (optional).

  Returns:
    A list of `num_splits` `Tensor` objects with the same type as `inputs`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUPartitionedOutput", name, inputs, "num_splits", num_splits,
        "partition_dim", partition_dim)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return tpu_partitioned_output_eager_fallback(
          inputs, num_splits=num_splits, partition_dim=partition_dim,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  num_splits = _execute.make_int(num_splits, "num_splits")
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUPartitionedOutput", inputs=inputs, num_splits=num_splits,
                                partition_dim=partition_dim, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "num_splits",
              _op._get_attr_int("num_splits"), "partition_dim",
              _op._get_attr_int("partition_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUPartitionedOutput", _inputs_flat, _attrs, _result)
  return _result

TPUPartitionedOutput = tf_export("raw_ops.TPUPartitionedOutput")(_ops.to_raw_op(tpu_partitioned_output))


def tpu_partitioned_output_eager_fallback(inputs, num_splits, partition_dim, name, ctx):
  num_splits = _execute.make_int(num_splits, "num_splits")
  if partition_dim is None:
    partition_dim = 0
  partition_dim = _execute.make_int(partition_dim, "partition_dim")
  _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [])
  _inputs_flat = [inputs]
  _attrs = ("T", _attr_T, "num_splits", num_splits, "partition_dim",
  partition_dim)
  _result = _execute.execute(b"TPUPartitionedOutput", num_splits,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUPartitionedOutput", _inputs_flat, _attrs, _result)
  return _result

