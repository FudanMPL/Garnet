"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: trt_ops.cc
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

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('create_trt_resource_handle')
def create_trt_resource_handle(resource_name, name=None):
  r"""TODO: add doc.

  Args:
    resource_name: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CreateTRTResourceHandle", name, "resource_name", resource_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_create_trt_resource_handle(
          (resource_name, name,), None)
      if _result is not NotImplemented:
        return _result
      return create_trt_resource_handle_eager_fallback(
          resource_name=resource_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            create_trt_resource_handle, (), dict(resource_name=resource_name,
                                                 name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_create_trt_resource_handle(
        (resource_name, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  resource_name = _execute.make_str(resource_name, "resource_name")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CreateTRTResourceHandle", resource_name=resource_name, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          create_trt_resource_handle, (), dict(resource_name=resource_name,
                                               name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("resource_name", _op.get_attr("resource_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CreateTRTResourceHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CreateTRTResourceHandle = tf_export("raw_ops.CreateTRTResourceHandle")(_ops.to_raw_op(create_trt_resource_handle))
_dispatcher_for_create_trt_resource_handle = create_trt_resource_handle._tf_type_based_dispatcher.Dispatch


def create_trt_resource_handle_eager_fallback(resource_name, name, ctx):
  resource_name = _execute.make_str(resource_name, "resource_name")
  _inputs_flat = []
  _attrs = ("resource_name", resource_name)
  _result = _execute.execute(b"CreateTRTResourceHandle", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CreateTRTResourceHandle", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('get_calibration_data_op')
def get_calibration_data_op(resource_name, name=None):
  r"""Returns calibration data for the given resource name

  Args:
    resource_name: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "GetCalibrationDataOp", name, resource_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_get_calibration_data_op(
          (resource_name, name,), None)
      if _result is not NotImplemented:
        return _result
      return get_calibration_data_op_eager_fallback(
          resource_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            get_calibration_data_op, (), dict(resource_name=resource_name,
                                              name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_get_calibration_data_op(
        (resource_name, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "GetCalibrationDataOp", resource_name=resource_name, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          get_calibration_data_op, (), dict(resource_name=resource_name,
                                            name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ()
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "GetCalibrationDataOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

GetCalibrationDataOp = tf_export("raw_ops.GetCalibrationDataOp")(_ops.to_raw_op(get_calibration_data_op))
_dispatcher_for_get_calibration_data_op = get_calibration_data_op._tf_type_based_dispatcher.Dispatch


def get_calibration_data_op_eager_fallback(resource_name, name, ctx):
  resource_name = _ops.convert_to_tensor(resource_name, _dtypes.string)
  _inputs_flat = [resource_name]
  _attrs = None
  _result = _execute.execute(b"GetCalibrationDataOp", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "GetCalibrationDataOp", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('initialize_trt_resource')
def initialize_trt_resource(resource_handle, filename, max_cached_engines_count=1, name=None):
  r"""TODO: add doc.

  Args:
    resource_handle: A `Tensor` of type `resource`.
    filename: A `Tensor` of type `string`.
    max_cached_engines_count: An optional `int`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "InitializeTRTResource", name, resource_handle, filename,
        "max_cached_engines_count", max_cached_engines_count)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_initialize_trt_resource(
          (resource_handle, filename, max_cached_engines_count, name,), None)
      if _result is not NotImplemented:
        return _result
      return initialize_trt_resource_eager_fallback(
          resource_handle, filename,
          max_cached_engines_count=max_cached_engines_count, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            initialize_trt_resource, (), dict(resource_handle=resource_handle,
                                              filename=filename,
                                              max_cached_engines_count=max_cached_engines_count,
                                              name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_initialize_trt_resource(
        (resource_handle, filename, max_cached_engines_count, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "InitializeTRTResource", resource_handle=resource_handle,
                                 filename=filename,
                                 max_cached_engines_count=max_cached_engines_count,
                                 name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          initialize_trt_resource, (), dict(resource_handle=resource_handle,
                                            filename=filename,
                                            max_cached_engines_count=max_cached_engines_count,
                                            name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
InitializeTRTResource = tf_export("raw_ops.InitializeTRTResource")(_ops.to_raw_op(initialize_trt_resource))
_dispatcher_for_initialize_trt_resource = initialize_trt_resource._tf_type_based_dispatcher.Dispatch


def initialize_trt_resource_eager_fallback(resource_handle, filename, max_cached_engines_count, name, ctx):
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [resource_handle, filename]
  _attrs = ("max_cached_engines_count", max_cached_engines_count)
  _result = _execute.execute(b"InitializeTRTResource", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('serialize_trt_resource')
def serialize_trt_resource(resource_name, filename, delete_resource=False, save_gpu_specific_engines=True, name=None):
  r"""TODO: add doc.

  Args:
    resource_name: A `Tensor` of type `string`.
    filename: A `Tensor` of type `string`.
    delete_resource: An optional `bool`. Defaults to `False`.
    save_gpu_specific_engines: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SerializeTRTResource", name, resource_name, filename,
        "delete_resource", delete_resource, "save_gpu_specific_engines",
        save_gpu_specific_engines)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_serialize_trt_resource(
          (resource_name, filename, delete_resource,
          save_gpu_specific_engines, name,), None)
      if _result is not NotImplemented:
        return _result
      return serialize_trt_resource_eager_fallback(
          resource_name, filename, delete_resource=delete_resource,
          save_gpu_specific_engines=save_gpu_specific_engines, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            serialize_trt_resource, (), dict(resource_name=resource_name,
                                             filename=filename,
                                             delete_resource=delete_resource,
                                             save_gpu_specific_engines=save_gpu_specific_engines,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_serialize_trt_resource(
        (resource_name, filename, delete_resource, save_gpu_specific_engines,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if delete_resource is None:
    delete_resource = False
  delete_resource = _execute.make_bool(delete_resource, "delete_resource")
  if save_gpu_specific_engines is None:
    save_gpu_specific_engines = True
  save_gpu_specific_engines = _execute.make_bool(save_gpu_specific_engines, "save_gpu_specific_engines")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SerializeTRTResource", resource_name=resource_name,
                                filename=filename,
                                delete_resource=delete_resource,
                                save_gpu_specific_engines=save_gpu_specific_engines,
                                name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          serialize_trt_resource, (), dict(resource_name=resource_name,
                                           filename=filename,
                                           delete_resource=delete_resource,
                                           save_gpu_specific_engines=save_gpu_specific_engines,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
SerializeTRTResource = tf_export("raw_ops.SerializeTRTResource")(_ops.to_raw_op(serialize_trt_resource))
_dispatcher_for_serialize_trt_resource = serialize_trt_resource._tf_type_based_dispatcher.Dispatch


def serialize_trt_resource_eager_fallback(resource_name, filename, delete_resource, save_gpu_specific_engines, name, ctx):
  if delete_resource is None:
    delete_resource = False
  delete_resource = _execute.make_bool(delete_resource, "delete_resource")
  if save_gpu_specific_engines is None:
    save_gpu_specific_engines = True
  save_gpu_specific_engines = _execute.make_bool(save_gpu_specific_engines, "save_gpu_specific_engines")
  resource_name = _ops.convert_to_tensor(resource_name, _dtypes.string)
  filename = _ops.convert_to_tensor(filename, _dtypes.string)
  _inputs_flat = [resource_name, filename]
  _attrs = ("delete_resource", delete_resource, "save_gpu_specific_engines",
  save_gpu_specific_engines)
  _result = _execute.execute(b"SerializeTRTResource", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('trt_engine_op')
def trt_engine_op(in_tensor, serialized_segment, OutT, workspace_size_bytes, precision_mode, segment_func="", input_shapes=[], output_shapes=[], max_cached_engines_count=1, max_batch_size=1, calibration_data="", use_calibration=True, segment_funcdef_name="", cached_engine_batches=[], fixed_input_size=True, static_engine=True, profile_strategy="", use_explicit_precision=False, name=None):
  r"""TODO: add doc.

  Args:
    in_tensor: A list of `Tensor` objects with types from: `bool`, `int8`, `half`, `float32`, `int32`, `resource`.
    serialized_segment: A `string`.
    OutT: A list of `tf.DTypes` from: `tf.bool, tf.int8, tf.half, tf.float32, tf.int32` that has length `>= 1`.
    workspace_size_bytes: An `int`.
    precision_mode: A `string` from: `"FP32", "FP16", "INT8"`.
    segment_func: An optional function decorated with @Defun. Defaults to `""`.
    input_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    max_cached_engines_count: An optional `int`. Defaults to `1`.
    max_batch_size: An optional `int`. Defaults to `1`.
    calibration_data: An optional `string`. Defaults to `""`.
    use_calibration: An optional `bool`. Defaults to `True`.
    segment_funcdef_name: An optional `string`. Defaults to `""`.
    cached_engine_batches: An optional list of `ints`. Defaults to `[]`.
    fixed_input_size: An optional `bool`. Defaults to `True`.
    static_engine: An optional `bool`. Defaults to `True`.
    profile_strategy: An optional `string`. Defaults to `""`.
    use_explicit_precision: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `OutT`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TRTEngineOp", name, in_tensor, "serialized_segment",
        serialized_segment, "segment_func", segment_func, "OutT", OutT,
        "input_shapes", input_shapes, "output_shapes", output_shapes,
        "max_cached_engines_count", max_cached_engines_count,
        "max_batch_size", max_batch_size, "workspace_size_bytes",
        workspace_size_bytes, "precision_mode", precision_mode,
        "calibration_data", calibration_data, "use_calibration",
        use_calibration, "segment_funcdef_name", segment_funcdef_name,
        "cached_engine_batches", cached_engine_batches, "fixed_input_size",
        fixed_input_size, "static_engine", static_engine, "profile_strategy",
        profile_strategy, "use_explicit_precision", use_explicit_precision)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_trt_engine_op(
          (in_tensor, serialized_segment, OutT, workspace_size_bytes,
          precision_mode, segment_func, input_shapes, output_shapes,
          max_cached_engines_count, max_batch_size, calibration_data,
          use_calibration, segment_funcdef_name, cached_engine_batches,
          fixed_input_size, static_engine, profile_strategy,
          use_explicit_precision, name,), None)
      if _result is not NotImplemented:
        return _result
      return trt_engine_op_eager_fallback(
          in_tensor, serialized_segment=serialized_segment,
          segment_func=segment_func, OutT=OutT, input_shapes=input_shapes,
          output_shapes=output_shapes,
          max_cached_engines_count=max_cached_engines_count,
          max_batch_size=max_batch_size,
          workspace_size_bytes=workspace_size_bytes,
          precision_mode=precision_mode, calibration_data=calibration_data,
          use_calibration=use_calibration,
          segment_funcdef_name=segment_funcdef_name,
          cached_engine_batches=cached_engine_batches,
          fixed_input_size=fixed_input_size, static_engine=static_engine,
          profile_strategy=profile_strategy,
          use_explicit_precision=use_explicit_precision, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            trt_engine_op, (), dict(in_tensor=in_tensor,
                                    serialized_segment=serialized_segment,
                                    OutT=OutT,
                                    workspace_size_bytes=workspace_size_bytes,
                                    precision_mode=precision_mode,
                                    segment_func=segment_func,
                                    input_shapes=input_shapes,
                                    output_shapes=output_shapes,
                                    max_cached_engines_count=max_cached_engines_count,
                                    max_batch_size=max_batch_size,
                                    calibration_data=calibration_data,
                                    use_calibration=use_calibration,
                                    segment_funcdef_name=segment_funcdef_name,
                                    cached_engine_batches=cached_engine_batches,
                                    fixed_input_size=fixed_input_size,
                                    static_engine=static_engine,
                                    profile_strategy=profile_strategy,
                                    use_explicit_precision=use_explicit_precision,
                                    name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_trt_engine_op(
        (in_tensor, serialized_segment, OutT, workspace_size_bytes,
        precision_mode, segment_func, input_shapes, output_shapes,
        max_cached_engines_count, max_batch_size, calibration_data,
        use_calibration, segment_funcdef_name, cached_engine_batches,
        fixed_input_size, static_engine, profile_strategy,
        use_explicit_precision, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  serialized_segment = _execute.make_str(serialized_segment, "serialized_segment")
  if not isinstance(OutT, (list, tuple)):
    raise TypeError(
        "Expected list for 'OutT' argument to "
        "'trt_engine_op' Op, not %r." % OutT)
  OutT = [_execute.make_type(_t, "OutT") for _t in OutT]
  workspace_size_bytes = _execute.make_int(workspace_size_bytes, "workspace_size_bytes")
  precision_mode = _execute.make_str(precision_mode, "precision_mode")
  if segment_func is None:
    segment_func = ""
  if input_shapes is None:
    input_shapes = []
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'trt_engine_op' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'trt_engine_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  if max_batch_size is None:
    max_batch_size = 1
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  if calibration_data is None:
    calibration_data = ""
  calibration_data = _execute.make_str(calibration_data, "calibration_data")
  if use_calibration is None:
    use_calibration = True
  use_calibration = _execute.make_bool(use_calibration, "use_calibration")
  if segment_funcdef_name is None:
    segment_funcdef_name = ""
  segment_funcdef_name = _execute.make_str(segment_funcdef_name, "segment_funcdef_name")
  if cached_engine_batches is None:
    cached_engine_batches = []
  if not isinstance(cached_engine_batches, (list, tuple)):
    raise TypeError(
        "Expected list for 'cached_engine_batches' argument to "
        "'trt_engine_op' Op, not %r." % cached_engine_batches)
  cached_engine_batches = [_execute.make_int(_i, "cached_engine_batches") for _i in cached_engine_batches]
  if fixed_input_size is None:
    fixed_input_size = True
  fixed_input_size = _execute.make_bool(fixed_input_size, "fixed_input_size")
  if static_engine is None:
    static_engine = True
  static_engine = _execute.make_bool(static_engine, "static_engine")
  if profile_strategy is None:
    profile_strategy = ""
  profile_strategy = _execute.make_str(profile_strategy, "profile_strategy")
  if use_explicit_precision is None:
    use_explicit_precision = False
  use_explicit_precision = _execute.make_bool(use_explicit_precision, "use_explicit_precision")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TRTEngineOp", in_tensor=in_tensor,
                       serialized_segment=serialized_segment, OutT=OutT,
                       workspace_size_bytes=workspace_size_bytes,
                       precision_mode=precision_mode,
                       segment_func=segment_func, input_shapes=input_shapes,
                       output_shapes=output_shapes,
                       max_cached_engines_count=max_cached_engines_count,
                       max_batch_size=max_batch_size,
                       calibration_data=calibration_data,
                       use_calibration=use_calibration,
                       segment_funcdef_name=segment_funcdef_name,
                       cached_engine_batches=cached_engine_batches,
                       fixed_input_size=fixed_input_size,
                       static_engine=static_engine,
                       profile_strategy=profile_strategy,
                       use_explicit_precision=use_explicit_precision,
                       name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          trt_engine_op, (), dict(in_tensor=in_tensor,
                                  serialized_segment=serialized_segment,
                                  OutT=OutT,
                                  workspace_size_bytes=workspace_size_bytes,
                                  precision_mode=precision_mode,
                                  segment_func=segment_func,
                                  input_shapes=input_shapes,
                                  output_shapes=output_shapes,
                                  max_cached_engines_count=max_cached_engines_count,
                                  max_batch_size=max_batch_size,
                                  calibration_data=calibration_data,
                                  use_calibration=use_calibration,
                                  segment_funcdef_name=segment_funcdef_name,
                                  cached_engine_batches=cached_engine_batches,
                                  fixed_input_size=fixed_input_size,
                                  static_engine=static_engine,
                                  profile_strategy=profile_strategy,
                                  use_explicit_precision=use_explicit_precision,
                                  name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("serialized_segment", _op.get_attr("serialized_segment"),
              "segment_func", _op.get_attr("segment_func"), "InT",
              _op.get_attr("InT"), "OutT", _op.get_attr("OutT"),
              "input_shapes", _op.get_attr("input_shapes"), "output_shapes",
              _op.get_attr("output_shapes"), "max_cached_engines_count",
              _op._get_attr_int("max_cached_engines_count"), "max_batch_size",
              _op._get_attr_int("max_batch_size"), "workspace_size_bytes",
              _op._get_attr_int("workspace_size_bytes"), "precision_mode",
              _op.get_attr("precision_mode"), "calibration_data",
              _op.get_attr("calibration_data"), "use_calibration",
              _op._get_attr_bool("use_calibration"), "segment_funcdef_name",
              _op.get_attr("segment_funcdef_name"), "cached_engine_batches",
              _op.get_attr("cached_engine_batches"), "fixed_input_size",
              _op._get_attr_bool("fixed_input_size"), "static_engine",
              _op._get_attr_bool("static_engine"), "profile_strategy",
              _op.get_attr("profile_strategy"), "use_explicit_precision",
              _op._get_attr_bool("use_explicit_precision"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TRTEngineOp", _inputs_flat, _attrs, _result)
  return _result

TRTEngineOp = tf_export("raw_ops.TRTEngineOp")(_ops.to_raw_op(trt_engine_op))
_dispatcher_for_trt_engine_op = trt_engine_op._tf_type_based_dispatcher.Dispatch


def trt_engine_op_eager_fallback(in_tensor, serialized_segment, OutT, workspace_size_bytes, precision_mode, segment_func, input_shapes, output_shapes, max_cached_engines_count, max_batch_size, calibration_data, use_calibration, segment_funcdef_name, cached_engine_batches, fixed_input_size, static_engine, profile_strategy, use_explicit_precision, name, ctx):
  serialized_segment = _execute.make_str(serialized_segment, "serialized_segment")
  if not isinstance(OutT, (list, tuple)):
    raise TypeError(
        "Expected list for 'OutT' argument to "
        "'trt_engine_op' Op, not %r." % OutT)
  OutT = [_execute.make_type(_t, "OutT") for _t in OutT]
  workspace_size_bytes = _execute.make_int(workspace_size_bytes, "workspace_size_bytes")
  precision_mode = _execute.make_str(precision_mode, "precision_mode")
  if segment_func is None:
    segment_func = ""
  if input_shapes is None:
    input_shapes = []
  if not isinstance(input_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_shapes' argument to "
        "'trt_engine_op' Op, not %r." % input_shapes)
  input_shapes = [_execute.make_shape(_s, "input_shapes") for _s in input_shapes]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'trt_engine_op' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if max_cached_engines_count is None:
    max_cached_engines_count = 1
  max_cached_engines_count = _execute.make_int(max_cached_engines_count, "max_cached_engines_count")
  if max_batch_size is None:
    max_batch_size = 1
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  if calibration_data is None:
    calibration_data = ""
  calibration_data = _execute.make_str(calibration_data, "calibration_data")
  if use_calibration is None:
    use_calibration = True
  use_calibration = _execute.make_bool(use_calibration, "use_calibration")
  if segment_funcdef_name is None:
    segment_funcdef_name = ""
  segment_funcdef_name = _execute.make_str(segment_funcdef_name, "segment_funcdef_name")
  if cached_engine_batches is None:
    cached_engine_batches = []
  if not isinstance(cached_engine_batches, (list, tuple)):
    raise TypeError(
        "Expected list for 'cached_engine_batches' argument to "
        "'trt_engine_op' Op, not %r." % cached_engine_batches)
  cached_engine_batches = [_execute.make_int(_i, "cached_engine_batches") for _i in cached_engine_batches]
  if fixed_input_size is None:
    fixed_input_size = True
  fixed_input_size = _execute.make_bool(fixed_input_size, "fixed_input_size")
  if static_engine is None:
    static_engine = True
  static_engine = _execute.make_bool(static_engine, "static_engine")
  if profile_strategy is None:
    profile_strategy = ""
  profile_strategy = _execute.make_str(profile_strategy, "profile_strategy")
  if use_explicit_precision is None:
    use_explicit_precision = False
  use_explicit_precision = _execute.make_bool(use_explicit_precision, "use_explicit_precision")
  _attr_InT, in_tensor = _execute.convert_to_mixed_eager_tensors(in_tensor, ctx)
  _inputs_flat = list(in_tensor)
  _attrs = ("serialized_segment", serialized_segment, "segment_func",
  segment_func, "InT", _attr_InT, "OutT", OutT, "input_shapes", input_shapes,
  "output_shapes", output_shapes, "max_cached_engines_count",
  max_cached_engines_count, "max_batch_size", max_batch_size,
  "workspace_size_bytes", workspace_size_bytes, "precision_mode",
  precision_mode, "calibration_data", calibration_data, "use_calibration",
  use_calibration, "segment_funcdef_name", segment_funcdef_name,
  "cached_engine_batches", cached_engine_batches, "fixed_input_size",
  fixed_input_size, "static_engine", static_engine, "profile_strategy",
  profile_strategy, "use_explicit_precision", use_explicit_precision)
  _result = _execute.execute(b"TRTEngineOp", len(OutT), inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TRTEngineOp", _inputs_flat, _attrs, _result)
  return _result

