"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: uniform_quant_ops.cc
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

def uniform_dequantize(input, scales, zero_points, Tout, quantization_min_val, quantization_max_val, quantization_axis=-1, name=None):
  r"""Perform dequantization on the quantized Tensor `input`.

  Given quantized `input` which was quantized using `scales` and `zero_points`, performs dequantization using the formula:
  dequantized_data = (quantized_data - zero_point) * scale.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
      Must be a Tensor of Tin.
    scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing original data that input represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing original data that input represents.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.float32`.
      The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
    quantization_min_val: An `int`.
      The quantization min value that was used when input was quantized.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    quantization_max_val: An `int`.
      The quantization max value that was used when input was quantized.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformDequantize", name, input, scales, zero_points, "Tout",
        Tout, "quantization_axis", quantization_axis, "quantization_min_val",
        quantization_min_val, "quantization_max_val", quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_dequantize_eager_fallback(
          input, scales, zero_points, Tout=Tout,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformDequantize", input=input, scales=scales,
                             zero_points=zero_points, Tout=Tout,
                             quantization_min_val=quantization_min_val,
                             quantization_max_val=quantization_max_val,
                             quantization_axis=quantization_axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformDequantize = tf_export("raw_ops.UniformDequantize")(_ops.to_raw_op(uniform_dequantize))


def uniform_dequantize_eager_fallback(input, scales, zero_points, Tout, quantization_min_val, quantization_max_val, quantization_axis, name, ctx):
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.qint32, ])
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [input, scales, zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "quantization_axis",
  quantization_axis, "quantization_min_val", quantization_min_val,
  "quantization_max_val", quantization_max_val)
  _result = _execute.execute(b"UniformDequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformDequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uniform_quantize(input, scales, zero_points, Tout, quantization_min_val, quantization_max_val, quantization_axis=-1, name=None):
  r"""Perform quantization on Tensor `input`.

  Given `input`, `scales` and `zero_points`, performs quantization using the formula:
  quantized_data = floor(input_data * (1.0f / scale) + 0.5f) + zero_point

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      Must be a Tensor of Tin.
    scales: A `Tensor` of type `float32`.
      The float value(s) to use as scale(s) to quantize `input`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as zero_point(s) to quantize `input`.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.float32
    quantization_min_val: An `int`.
      The quantization min value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    quantization_max_val: An `int`.
      The quantization max value to quantize `input`.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantize", name, input, scales, zero_points, "Tout",
        Tout, "quantization_axis", quantization_axis, "quantization_min_val",
        quantization_min_val, "quantization_max_val", quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantize_eager_fallback(
          input, scales, zero_points, Tout=Tout,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantize", input=input, scales=scales,
                           zero_points=zero_points, Tout=Tout,
                           quantization_min_val=quantization_min_val,
                           quantization_max_val=quantization_max_val,
                           quantization_axis=quantization_axis, name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantize = tf_export("raw_ops.UniformQuantize")(_ops.to_raw_op(uniform_quantize))


def uniform_quantize_eager_fallback(input, scales, zero_points, Tout, quantization_min_val, quantization_max_val, quantization_axis, name, ctx):
  Tout = _execute.make_type(Tout, "Tout")
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, ])
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [input, scales, zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "quantization_axis",
  quantization_axis, "quantization_min_val", quantization_min_val,
  "quantization_max_val", quantization_max_val)
  _result = _execute.execute(b"UniformQuantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uniform_quantized_clip_by_value(operand, min, max, scales, zero_points, quantization_min_val, quantization_max_val, quantization_axis=-1, name=None):
  r"""Perform clip by value on the quantized Tensor `operand`.

  Given quantized `operand` which was quantized using `scales` and `zero_points`, performs clip by value using `min` and `max` values.
  If quantization_axis is -1 (per-tensor quantized), the entire operand is clipped using scalar min, max.
  Otherwise (per-channel quantized), the clipping is also done per-channel.

  Args:
    operand: A `Tensor`. Must be one of the following types: `qint32`.
      Must be a Tensor of T.
    min: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    max: A `Tensor`. Must have the same type as `operand`.
      The min value(s) to clip operand. Must be a Tensor of T.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing `operand`, `min` and `max`.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
    zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing `operand`, `min` and `max`.
      Same shape condition as scales.
    quantization_min_val: An `int`.
      The quantization min value that was used when operand was quantized.
    quantization_max_val: An `int`.
      The quantization max value that was used when operand was quantized.
    quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, operand.dims()).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `operand`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedClipByValue", name, operand, min, max, scales,
        zero_points, "quantization_axis", quantization_axis,
        "quantization_min_val", quantization_min_val, "quantization_max_val",
        quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_clip_by_value_eager_fallback(
          operand, min, max, scales, zero_points,
          quantization_axis=quantization_axis,
          quantization_min_val=quantization_min_val,
          quantization_max_val=quantization_max_val, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedClipByValue", operand=operand, min=min, max=max,
                                       scales=scales, zero_points=zero_points,
                                       quantization_min_val=quantization_min_val,
                                       quantization_max_val=quantization_max_val,
                                       quantization_axis=quantization_axis,
                                       name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("T", _op._get_attr_type("T"), "quantization_axis",
              _op._get_attr_int("quantization_axis"), "quantization_min_val",
              _op._get_attr_int("quantization_min_val"),
              "quantization_max_val",
              _op._get_attr_int("quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedClipByValue = tf_export("raw_ops.UniformQuantizedClipByValue")(_ops.to_raw_op(uniform_quantized_clip_by_value))


def uniform_quantized_clip_by_value_eager_fallback(operand, min, max, scales, zero_points, quantization_min_val, quantization_max_val, quantization_axis, name, ctx):
  quantization_min_val = _execute.make_int(quantization_min_val, "quantization_min_val")
  quantization_max_val = _execute.make_int(quantization_max_val, "quantization_max_val")
  if quantization_axis is None:
    quantization_axis = -1
  quantization_axis = _execute.make_int(quantization_axis, "quantization_axis")
  _attr_T, _inputs_T = _execute.args_to_matching_eager([operand, min, max], ctx, [_dtypes.qint32, ])
  (operand, min, max) = _inputs_T
  scales = _ops.convert_to_tensor(scales, _dtypes.float32)
  zero_points = _ops.convert_to_tensor(zero_points, _dtypes.int32)
  _inputs_flat = [operand, min, max, scales, zero_points]
  _attrs = ("T", _attr_T, "quantization_axis", quantization_axis,
  "quantization_min_val", quantization_min_val, "quantization_max_val",
  quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedClipByValue", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedClipByValue", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uniform_quantized_dot(lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, Tout, lhs_quantization_min_val, lhs_quantization_max_val, rhs_quantization_min_val, rhs_quantization_max_val, output_quantization_min_val, output_quantization_max_val, lhs_quantization_axis=-1, rhs_quantization_axis=-1, output_quantization_axis=-1, name=None):
  r"""Perform quantized dot of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.

  Given quantized `lhs` and quantized `rhs`, performs quantized dot on `lhs` and `rhs` to make quantized `output`.
  `lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
  `lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).
  `output` is also quantized, using the same formula.
  If `rhs` is per-tensor quantized, `output` must be also per-tensor quantized.

  Args:
    lhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a 2D Tensor of Tin.
    rhs: A `Tensor`. Must have the same type as `lhs`.
      Must be a 2D Tensor of Tin.
    lhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that lhs represents.
      Must be a scalar Tensor (lhs supports only per-tensor quantization).
    lhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that lhs represents.
      Same shape condition as lhs_scales.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that rhs represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that rhs represents.
      Same shape condition as rhs_scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as scales when quantizing original data that output represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (output.dim_size(1),) (per-channel quantization).
      If rhs is per-tensor quantized, output must be also per-tensor quantized.
      This means that if rhs_scales and rhs_zero_points are scalar Tensors, output_scales and output_zero_points must be scalar Tensors as well.
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that output represents.
      Same shape condition as rhs_scales.
    Tout: A `tf.DType` from: `tf.qint32`. The type of output Tensor.
    lhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in lhs.
      For example, if Tin is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    lhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Tin is qint8, this must be set to 127.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to 127.
    output_quantization_min_val: An `int`.
      The min value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    output_quantization_max_val: An `int`.
      The max value of the quantized data stored in output.
      For example, if Tout is qint8, this must be set to 127.
    lhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op lhs, only per-tensor quantization is supported.
      Thus, this attribute must be set to -1. Other values are rejected.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op output, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedDot", name, lhs, rhs, lhs_scales,
        lhs_zero_points, rhs_scales, rhs_zero_points, output_scales,
        output_zero_points, "Tout", Tout, "lhs_quantization_axis",
        lhs_quantization_axis, "lhs_quantization_min_val",
        lhs_quantization_min_val, "lhs_quantization_max_val",
        lhs_quantization_max_val, "rhs_quantization_axis",
        rhs_quantization_axis, "rhs_quantization_min_val",
        rhs_quantization_min_val, "rhs_quantization_max_val",
        rhs_quantization_max_val, "output_quantization_axis",
        output_quantization_axis, "output_quantization_min_val",
        output_quantization_min_val, "output_quantization_max_val",
        output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_dot_eager_fallback(
          lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points,
          output_scales, output_zero_points, Tout=Tout,
          lhs_quantization_axis=lhs_quantization_axis,
          lhs_quantization_min_val=lhs_quantization_min_val,
          lhs_quantization_max_val=lhs_quantization_max_val,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedDot", lhs=lhs, rhs=rhs, lhs_scales=lhs_scales,
                               lhs_zero_points=lhs_zero_points,
                               rhs_scales=rhs_scales,
                               rhs_zero_points=rhs_zero_points,
                               output_scales=output_scales,
                               output_zero_points=output_zero_points,
                               Tout=Tout,
                               lhs_quantization_min_val=lhs_quantization_min_val,
                               lhs_quantization_max_val=lhs_quantization_max_val,
                               rhs_quantization_min_val=rhs_quantization_min_val,
                               rhs_quantization_max_val=rhs_quantization_max_val,
                               output_quantization_min_val=output_quantization_min_val,
                               output_quantization_max_val=output_quantization_max_val,
                               lhs_quantization_axis=lhs_quantization_axis,
                               rhs_quantization_axis=rhs_quantization_axis,
                               output_quantization_axis=output_quantization_axis,
                               name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "lhs_quantization_axis",
              _op._get_attr_int("lhs_quantization_axis"),
              "lhs_quantization_min_val",
              _op._get_attr_int("lhs_quantization_min_val"),
              "lhs_quantization_max_val",
              _op._get_attr_int("lhs_quantization_max_val"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedDot = tf_export("raw_ops.UniformQuantizedDot")(_ops.to_raw_op(uniform_quantized_dot))


def uniform_quantized_dot_eager_fallback(lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points, Tout, lhs_quantization_min_val, lhs_quantization_max_val, rhs_quantization_min_val, rhs_quantization_max_val, output_quantization_min_val, output_quantization_max_val, lhs_quantization_axis, rhs_quantization_axis, output_quantization_axis, name, ctx):
  Tout = _execute.make_type(Tout, "Tout")
  lhs_quantization_min_val = _execute.make_int(lhs_quantization_min_val, "lhs_quantization_min_val")
  lhs_quantization_max_val = _execute.make_int(lhs_quantization_max_val, "lhs_quantization_max_val")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if lhs_quantization_axis is None:
    lhs_quantization_axis = -1
  lhs_quantization_axis = _execute.make_int(lhs_quantization_axis, "lhs_quantization_axis")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_Tin, _inputs_Tin = _execute.args_to_matching_eager([lhs, rhs], ctx, [_dtypes.qint8, ])
  (lhs, rhs) = _inputs_Tin
  lhs_scales = _ops.convert_to_tensor(lhs_scales, _dtypes.float32)
  lhs_zero_points = _ops.convert_to_tensor(lhs_zero_points, _dtypes.int32)
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, lhs_scales, lhs_zero_points, rhs_scales, rhs_zero_points, output_scales, output_zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "lhs_quantization_axis",
  lhs_quantization_axis, "lhs_quantization_min_val", lhs_quantization_min_val,
  "lhs_quantization_max_val", lhs_quantization_max_val,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedDot", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedDot", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uniform_quantized_dot_hybrid(lhs, rhs, rhs_scales, rhs_zero_points, Tout, rhs_quantization_min_val, rhs_quantization_max_val, rhs_quantization_axis=-1, name=None):
  r"""Perform hybrid quantized dot of float Tensor `lhs` and quantized Tensor `rhs`.

  Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`, and then performs quantized dot on quantized lhs and `rhs`.
  The internal quantization on `lhs` is a quantization to qint8, dynamic range, per-batch (per-axis along axis 0), asymmetric, and not narrow range (the range is [-128, 127]).
  `lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
  `rhs` must be quantized Tensor, where its data value is quantized using the formula:
  quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).

  Args:
    lhs: A `Tensor`. Must be one of the following types: `float32`.
      Must be a 2D Tensor of Tlhs.
    rhs: A `Tensor`. Must be one of the following types: `qint8`.
      Must be a 2D Tensor of Trhs.
    rhs_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale when quantizing original data that rhs represents.
      Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
    rhs_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point when quantizing original data that rhs represents.
      Same shape condition as rhs_scales.
    Tout: A `tf.DType` from: `tf.float32`. The type of output Tensor.
    rhs_quantization_min_val: An `int`.
      The min value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
    rhs_quantization_max_val: An `int`.
      The max value of the quantized data stored in rhs.
      For example, if Trhs is qint8, this must be set to 127.
    rhs_quantization_axis: An optional `int`. Defaults to `-1`.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization.
      For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
      Thus, this attribute must be set to -1 or 1. Other values are rejected.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformQuantizedDotHybrid", name, lhs, rhs, rhs_scales,
        rhs_zero_points, "Tout", Tout, "rhs_quantization_axis",
        rhs_quantization_axis, "rhs_quantization_min_val",
        rhs_quantization_min_val, "rhs_quantization_max_val",
        rhs_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_quantized_dot_hybrid_eager_fallback(
          lhs, rhs, rhs_scales, rhs_zero_points, Tout=Tout,
          rhs_quantization_axis=rhs_quantization_axis,
          rhs_quantization_min_val=rhs_quantization_min_val,
          rhs_quantization_max_val=rhs_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformQuantizedDotHybrid", lhs=lhs, rhs=rhs, rhs_scales=rhs_scales,
                                     rhs_zero_points=rhs_zero_points,
                                     Tout=Tout,
                                     rhs_quantization_min_val=rhs_quantization_min_val,
                                     rhs_quantization_max_val=rhs_quantization_max_val,
                                     rhs_quantization_axis=rhs_quantization_axis,
                                     name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tlhs", _op._get_attr_type("Tlhs"), "Trhs",
              _op._get_attr_type("Trhs"), "Tout", _op._get_attr_type("Tout"),
              "rhs_quantization_axis",
              _op._get_attr_int("rhs_quantization_axis"),
              "rhs_quantization_min_val",
              _op._get_attr_int("rhs_quantization_min_val"),
              "rhs_quantization_max_val",
              _op._get_attr_int("rhs_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformQuantizedDotHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformQuantizedDotHybrid = tf_export("raw_ops.UniformQuantizedDotHybrid")(_ops.to_raw_op(uniform_quantized_dot_hybrid))


def uniform_quantized_dot_hybrid_eager_fallback(lhs, rhs, rhs_scales, rhs_zero_points, Tout, rhs_quantization_min_val, rhs_quantization_max_val, rhs_quantization_axis, name, ctx):
  Tout = _execute.make_type(Tout, "Tout")
  rhs_quantization_min_val = _execute.make_int(rhs_quantization_min_val, "rhs_quantization_min_val")
  rhs_quantization_max_val = _execute.make_int(rhs_quantization_max_val, "rhs_quantization_max_val")
  if rhs_quantization_axis is None:
    rhs_quantization_axis = -1
  rhs_quantization_axis = _execute.make_int(rhs_quantization_axis, "rhs_quantization_axis")
  _attr_Tlhs, (lhs,) = _execute.args_to_matching_eager([lhs], ctx, [_dtypes.float32, ])
  _attr_Trhs, (rhs,) = _execute.args_to_matching_eager([rhs], ctx, [_dtypes.qint8, ])
  rhs_scales = _ops.convert_to_tensor(rhs_scales, _dtypes.float32)
  rhs_zero_points = _ops.convert_to_tensor(rhs_zero_points, _dtypes.int32)
  _inputs_flat = [lhs, rhs, rhs_scales, rhs_zero_points]
  _attrs = ("Tlhs", _attr_Tlhs, "Trhs", _attr_Trhs, "Tout", Tout,
  "rhs_quantization_axis", rhs_quantization_axis, "rhs_quantization_min_val",
  rhs_quantization_min_val, "rhs_quantization_max_val",
  rhs_quantization_max_val)
  _result = _execute.execute(b"UniformQuantizedDotHybrid", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformQuantizedDotHybrid", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def uniform_requantize(input, input_scales, input_zero_points, output_scales, output_zero_points, Tout, input_quantization_min_val, input_quantization_max_val, output_quantization_min_val, output_quantization_max_val, input_quantization_axis=-1, output_quantization_axis=-1, name=None):
  r"""Given quantized tensor `input`, requantize it with new quantization parameters.

  Given quantized tensor `input`, which was quantized using {input_scales, input_zero_points, input_quantization_axis, input_quantization_min_val, input_quantization_max_val},
  requantize it to a tensor, which is quantized using {output_scales, output_zero_points, output_quantization_axis, output_quantization_min_val, output_quantization_max_val}.
  The requantization is done by using the formula:
  output_quantized_data = clip(
    (input_quantized_data - input_zero_point) * (input_scale / output_scale) + output_zero_point,
    output_quantization_min_val,
    output_quantization_max_val)

  Per-tensor and per-axis quantization supported cases are followings:
  * per-tensor -> per-tensor
  * per-tensor -> per-axis
  * per-axis -> per-axis where input_quantization_axis equals output_quantization_axis.
  i.e. At least one among input_quantization_axis and output_quantization_axis must be -1, or two must be equal.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
      Must be a Tensor of Tin.
    input_scales: A `Tensor` of type `float32`.
      The float value(s) used as scale(s) when quantizing original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    input_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) used as zero_point(s) when quantizing original data that `input` represents.
      Same shape condition as scales.
    output_scales: A `Tensor` of type `float32`.
      The float value(s) to use as new scale(s) to quantize original data that `input` represents.
      Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
    output_zero_points: A `Tensor` of type `int32`.
      The int32 value(s) to use as new zero_point(s) to quantize original data that `input` represents.
      Same shape condition as scales.
    Tout: A `tf.DType` from: `tf.qint8, tf.qint32`.
      The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
    input_quantization_min_val: An `int`.
      The quantization min value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
      `(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
      For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
    input_quantization_max_val: An `int`.
      The quantization max value that was used when quantizing original data that `input` represents.
      The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
      `(Tout max)` for both narrow range and not narrow range.
      For example, if Tin is qint8, this is set to 127.
    output_quantization_min_val: An `int`.
      The new quantization min value to quantize original data that `input` represents.
    output_quantization_max_val: An `int`.
      The new quantization max value to quantize original data that `input` represents.
    input_quantization_axis: An optional `int`. Defaults to `-1`.
      The quantization axis that was used when quantizing original data that `input` represents.
      Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
      If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
    output_quantization_axis: An optional `int`. Defaults to `-1`.
      The new quantization axis to use to quantize original data that `input` represents.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "UniformRequantize", name, input, input_scales,
        input_zero_points, output_scales, output_zero_points, "Tout", Tout,
        "input_quantization_axis", input_quantization_axis,
        "input_quantization_min_val", input_quantization_min_val,
        "input_quantization_max_val", input_quantization_max_val,
        "output_quantization_axis", output_quantization_axis,
        "output_quantization_min_val", output_quantization_min_val,
        "output_quantization_max_val", output_quantization_max_val)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return uniform_requantize_eager_fallback(
          input, input_scales, input_zero_points, output_scales,
          output_zero_points, Tout=Tout,
          input_quantization_axis=input_quantization_axis,
          input_quantization_min_val=input_quantization_min_val,
          input_quantization_max_val=input_quantization_max_val,
          output_quantization_axis=output_quantization_axis,
          output_quantization_min_val=output_quantization_min_val,
          output_quantization_max_val=output_quantization_max_val, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
  # Add nodes to the TensorFlow graph.
  Tout = _execute.make_type(Tout, "Tout")
  input_quantization_min_val = _execute.make_int(input_quantization_min_val, "input_quantization_min_val")
  input_quantization_max_val = _execute.make_int(input_quantization_max_val, "input_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if input_quantization_axis is None:
    input_quantization_axis = -1
  input_quantization_axis = _execute.make_int(input_quantization_axis, "input_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "UniformRequantize", input=input, input_scales=input_scales,
                             input_zero_points=input_zero_points,
                             output_scales=output_scales,
                             output_zero_points=output_zero_points, Tout=Tout,
                             input_quantization_min_val=input_quantization_min_val,
                             input_quantization_max_val=input_quantization_max_val,
                             output_quantization_min_val=output_quantization_min_val,
                             output_quantization_max_val=output_quantization_max_val,
                             input_quantization_axis=input_quantization_axis,
                             output_quantization_axis=output_quantization_axis,
                             name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("Tin", _op._get_attr_type("Tin"), "Tout",
              _op._get_attr_type("Tout"), "input_quantization_axis",
              _op._get_attr_int("input_quantization_axis"),
              "input_quantization_min_val",
              _op._get_attr_int("input_quantization_min_val"),
              "input_quantization_max_val",
              _op._get_attr_int("input_quantization_max_val"),
              "output_quantization_axis",
              _op._get_attr_int("output_quantization_axis"),
              "output_quantization_min_val",
              _op._get_attr_int("output_quantization_min_val"),
              "output_quantization_max_val",
              _op._get_attr_int("output_quantization_max_val"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "UniformRequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

UniformRequantize = tf_export("raw_ops.UniformRequantize")(_ops.to_raw_op(uniform_requantize))


def uniform_requantize_eager_fallback(input, input_scales, input_zero_points, output_scales, output_zero_points, Tout, input_quantization_min_val, input_quantization_max_val, output_quantization_min_val, output_quantization_max_val, input_quantization_axis, output_quantization_axis, name, ctx):
  Tout = _execute.make_type(Tout, "Tout")
  input_quantization_min_val = _execute.make_int(input_quantization_min_val, "input_quantization_min_val")
  input_quantization_max_val = _execute.make_int(input_quantization_max_val, "input_quantization_max_val")
  output_quantization_min_val = _execute.make_int(output_quantization_min_val, "output_quantization_min_val")
  output_quantization_max_val = _execute.make_int(output_quantization_max_val, "output_quantization_max_val")
  if input_quantization_axis is None:
    input_quantization_axis = -1
  input_quantization_axis = _execute.make_int(input_quantization_axis, "input_quantization_axis")
  if output_quantization_axis is None:
    output_quantization_axis = -1
  output_quantization_axis = _execute.make_int(output_quantization_axis, "output_quantization_axis")
  _attr_Tin, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.qint32, ])
  input_scales = _ops.convert_to_tensor(input_scales, _dtypes.float32)
  input_zero_points = _ops.convert_to_tensor(input_zero_points, _dtypes.int32)
  output_scales = _ops.convert_to_tensor(output_scales, _dtypes.float32)
  output_zero_points = _ops.convert_to_tensor(output_zero_points, _dtypes.int32)
  _inputs_flat = [input, input_scales, input_zero_points, output_scales, output_zero_points]
  _attrs = ("Tin", _attr_Tin, "Tout", Tout, "input_quantization_axis",
  input_quantization_axis, "input_quantization_min_val",
  input_quantization_min_val, "input_quantization_max_val",
  input_quantization_max_val, "output_quantization_axis",
  output_quantization_axis, "output_quantization_min_val",
  output_quantization_min_val, "output_quantization_max_val",
  output_quantization_max_val)
  _result = _execute.execute(b"UniformRequantize", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "UniformRequantize", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

