"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_tpu_embedding_ops.cc
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
@tf_export('collate_tpu_embedding_memory')
def collate_tpu_embedding_memory(memory_configs, name=None):
  r"""TODO: add doc.

  Args:
    memory_configs: A list of at least 1 `Tensor` objects with type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "CollateTPUEmbeddingMemory", name, memory_configs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_collate_tpu_embedding_memory(
          (memory_configs, name,), None)
      if _result is not NotImplemented:
        return _result
      return collate_tpu_embedding_memory_eager_fallback(
          memory_configs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            collate_tpu_embedding_memory, (), dict(memory_configs=memory_configs,
                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_collate_tpu_embedding_memory(
        (memory_configs, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(memory_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'memory_configs' argument to "
        "'collate_tpu_embedding_memory' Op, not %r." % memory_configs)
  _attr_N = len(memory_configs)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CollateTPUEmbeddingMemory", memory_configs=memory_configs, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          collate_tpu_embedding_memory, (), dict(memory_configs=memory_configs,
                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("N", _op._get_attr_int("N"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "CollateTPUEmbeddingMemory", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

CollateTPUEmbeddingMemory = tf_export("raw_ops.CollateTPUEmbeddingMemory")(_ops.to_raw_op(collate_tpu_embedding_memory))
_dispatcher_for_collate_tpu_embedding_memory = collate_tpu_embedding_memory._tf_type_based_dispatcher.Dispatch


def collate_tpu_embedding_memory_eager_fallback(memory_configs, name, ctx):
  if not isinstance(memory_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'memory_configs' argument to "
        "'collate_tpu_embedding_memory' Op, not %r." % memory_configs)
  _attr_N = len(memory_configs)
  memory_configs = _ops.convert_n_to_tensor(memory_configs, _dtypes.string)
  _inputs_flat = list(memory_configs)
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"CollateTPUEmbeddingMemory", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "CollateTPUEmbeddingMemory", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('configure_distributed_tpu')
def configure_distributed_tpu(embedding_config="", tpu_embedding_config="", is_global_init=False, enable_whole_mesh_compilations=False, compilation_failure_closes_chips=True, tpu_cancellation_closes_chips=0, name=None):
  r"""TODO: add doc.

  Args:
    embedding_config: An optional `string`. Defaults to `""`.
    tpu_embedding_config: An optional `string`. Defaults to `""`.
    is_global_init: An optional `bool`. Defaults to `False`.
    enable_whole_mesh_compilations: An optional `bool`. Defaults to `False`.
    compilation_failure_closes_chips: An optional `bool`. Defaults to `True`.
    tpu_cancellation_closes_chips: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureDistributedTPU", name, "embedding_config",
        embedding_config, "tpu_embedding_config", tpu_embedding_config,
        "is_global_init", is_global_init, "enable_whole_mesh_compilations",
        enable_whole_mesh_compilations, "compilation_failure_closes_chips",
        compilation_failure_closes_chips, "tpu_cancellation_closes_chips",
        tpu_cancellation_closes_chips)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_configure_distributed_tpu(
          (embedding_config, tpu_embedding_config, is_global_init,
          enable_whole_mesh_compilations, compilation_failure_closes_chips,
          tpu_cancellation_closes_chips, name,), None)
      if _result is not NotImplemented:
        return _result
      return configure_distributed_tpu_eager_fallback(
          embedding_config=embedding_config,
          tpu_embedding_config=tpu_embedding_config,
          is_global_init=is_global_init,
          enable_whole_mesh_compilations=enable_whole_mesh_compilations,
          compilation_failure_closes_chips=compilation_failure_closes_chips,
          tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            configure_distributed_tpu, (), dict(embedding_config=embedding_config,
                                                tpu_embedding_config=tpu_embedding_config,
                                                is_global_init=is_global_init,
                                                enable_whole_mesh_compilations=enable_whole_mesh_compilations,
                                                compilation_failure_closes_chips=compilation_failure_closes_chips,
                                                tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_configure_distributed_tpu(
        (embedding_config, tpu_embedding_config, is_global_init,
        enable_whole_mesh_compilations, compilation_failure_closes_chips,
        tpu_cancellation_closes_chips, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if embedding_config is None:
    embedding_config = ""
  embedding_config = _execute.make_str(embedding_config, "embedding_config")
  if tpu_embedding_config is None:
    tpu_embedding_config = ""
  tpu_embedding_config = _execute.make_str(tpu_embedding_config, "tpu_embedding_config")
  if is_global_init is None:
    is_global_init = False
  is_global_init = _execute.make_bool(is_global_init, "is_global_init")
  if enable_whole_mesh_compilations is None:
    enable_whole_mesh_compilations = False
  enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, "enable_whole_mesh_compilations")
  if compilation_failure_closes_chips is None:
    compilation_failure_closes_chips = True
  compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, "compilation_failure_closes_chips")
  if tpu_cancellation_closes_chips is None:
    tpu_cancellation_closes_chips = 0
  tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, "tpu_cancellation_closes_chips")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureDistributedTPU", embedding_config=embedding_config,
                                   tpu_embedding_config=tpu_embedding_config,
                                   is_global_init=is_global_init,
                                   enable_whole_mesh_compilations=enable_whole_mesh_compilations,
                                   compilation_failure_closes_chips=compilation_failure_closes_chips,
                                   tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          configure_distributed_tpu, (), dict(embedding_config=embedding_config,
                                              tpu_embedding_config=tpu_embedding_config,
                                              is_global_init=is_global_init,
                                              enable_whole_mesh_compilations=enable_whole_mesh_compilations,
                                              compilation_failure_closes_chips=compilation_failure_closes_chips,
                                              tpu_cancellation_closes_chips=tpu_cancellation_closes_chips,
                                              name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("embedding_config", _op.get_attr("embedding_config"),
              "tpu_embedding_config", _op.get_attr("tpu_embedding_config"),
              "is_global_init", _op._get_attr_bool("is_global_init"),
              "enable_whole_mesh_compilations",
              _op._get_attr_bool("enable_whole_mesh_compilations"),
              "compilation_failure_closes_chips",
              _op._get_attr_bool("compilation_failure_closes_chips"),
              "tpu_cancellation_closes_chips",
              _op._get_attr_int("tpu_cancellation_closes_chips"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConfigureDistributedTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConfigureDistributedTPU = tf_export("raw_ops.ConfigureDistributedTPU")(_ops.to_raw_op(configure_distributed_tpu))
_dispatcher_for_configure_distributed_tpu = configure_distributed_tpu._tf_type_based_dispatcher.Dispatch


def configure_distributed_tpu_eager_fallback(embedding_config, tpu_embedding_config, is_global_init, enable_whole_mesh_compilations, compilation_failure_closes_chips, tpu_cancellation_closes_chips, name, ctx):
  if embedding_config is None:
    embedding_config = ""
  embedding_config = _execute.make_str(embedding_config, "embedding_config")
  if tpu_embedding_config is None:
    tpu_embedding_config = ""
  tpu_embedding_config = _execute.make_str(tpu_embedding_config, "tpu_embedding_config")
  if is_global_init is None:
    is_global_init = False
  is_global_init = _execute.make_bool(is_global_init, "is_global_init")
  if enable_whole_mesh_compilations is None:
    enable_whole_mesh_compilations = False
  enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, "enable_whole_mesh_compilations")
  if compilation_failure_closes_chips is None:
    compilation_failure_closes_chips = True
  compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, "compilation_failure_closes_chips")
  if tpu_cancellation_closes_chips is None:
    tpu_cancellation_closes_chips = 0
  tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, "tpu_cancellation_closes_chips")
  _inputs_flat = []
  _attrs = ("embedding_config", embedding_config, "tpu_embedding_config",
  tpu_embedding_config, "is_global_init", is_global_init,
  "enable_whole_mesh_compilations", enable_whole_mesh_compilations,
  "compilation_failure_closes_chips", compilation_failure_closes_chips,
  "tpu_cancellation_closes_chips", tpu_cancellation_closes_chips)
  _result = _execute.execute(b"ConfigureDistributedTPU", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConfigureDistributedTPU", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('configure_tpu_embedding')
def configure_tpu_embedding(config, name=None):
  r"""TODO: add doc.

  Args:
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureTPUEmbedding", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_configure_tpu_embedding(
          (config, name,), None)
      if _result is not NotImplemented:
        return _result
      return configure_tpu_embedding_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            configure_tpu_embedding, (), dict(config=config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_configure_tpu_embedding(
        (config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureTPUEmbedding", config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          configure_tpu_embedding, (), dict(config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ConfigureTPUEmbedding = tf_export("raw_ops.ConfigureTPUEmbedding")(_ops.to_raw_op(configure_tpu_embedding))
_dispatcher_for_configure_tpu_embedding = configure_tpu_embedding._tf_type_based_dispatcher.Dispatch


def configure_tpu_embedding_eager_fallback(config, name, ctx):
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"ConfigureTPUEmbedding", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('configure_tpu_embedding_host')
def configure_tpu_embedding_host(common_config, memory_config, config, name=None):
  r"""TODO: add doc.

  Args:
    common_config: A `Tensor` of type `string`.
    memory_config: A `Tensor` of type `string`.
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureTPUEmbeddingHost", name, common_config, memory_config,
        "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_configure_tpu_embedding_host(
          (common_config, memory_config, config, name,), None)
      if _result is not NotImplemented:
        return _result
      return configure_tpu_embedding_host_eager_fallback(
          common_config, memory_config, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            configure_tpu_embedding_host, (), dict(common_config=common_config,
                                                   memory_config=memory_config,
                                                   config=config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_configure_tpu_embedding_host(
        (common_config, memory_config, config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureTPUEmbeddingHost", common_config=common_config,
                                     memory_config=memory_config,
                                     config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          configure_tpu_embedding_host, (), dict(common_config=common_config,
                                                 memory_config=memory_config,
                                                 config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ConfigureTPUEmbeddingHost", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConfigureTPUEmbeddingHost = tf_export("raw_ops.ConfigureTPUEmbeddingHost")(_ops.to_raw_op(configure_tpu_embedding_host))
_dispatcher_for_configure_tpu_embedding_host = configure_tpu_embedding_host._tf_type_based_dispatcher.Dispatch


def configure_tpu_embedding_host_eager_fallback(common_config, memory_config, config, name, ctx):
  config = _execute.make_str(config, "config")
  common_config = _ops.convert_to_tensor(common_config, _dtypes.string)
  memory_config = _ops.convert_to_tensor(memory_config, _dtypes.string)
  _inputs_flat = [common_config, memory_config]
  _attrs = ("config", config)
  _result = _execute.execute(b"ConfigureTPUEmbeddingHost", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConfigureTPUEmbeddingHost", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('configure_tpu_embedding_memory')
def configure_tpu_embedding_memory(common_config, name=None):
  r"""TODO: add doc.

  Args:
    common_config: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConfigureTPUEmbeddingMemory", name, common_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_configure_tpu_embedding_memory(
          (common_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return configure_tpu_embedding_memory_eager_fallback(
          common_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            configure_tpu_embedding_memory, (), dict(common_config=common_config,
                                                     name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_configure_tpu_embedding_memory(
        (common_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConfigureTPUEmbeddingMemory", common_config=common_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          configure_tpu_embedding_memory, (), dict(common_config=common_config,
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
        "ConfigureTPUEmbeddingMemory", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ConfigureTPUEmbeddingMemory = tf_export("raw_ops.ConfigureTPUEmbeddingMemory")(_ops.to_raw_op(configure_tpu_embedding_memory))
_dispatcher_for_configure_tpu_embedding_memory = configure_tpu_embedding_memory._tf_type_based_dispatcher.Dispatch


def configure_tpu_embedding_memory_eager_fallback(common_config, name, ctx):
  common_config = _ops.convert_to_tensor(common_config, _dtypes.string)
  _inputs_flat = [common_config]
  _attrs = None
  _result = _execute.execute(b"ConfigureTPUEmbeddingMemory", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ConfigureTPUEmbeddingMemory", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('connect_tpu_embedding_hosts')
def connect_tpu_embedding_hosts(network_configs, name=None):
  r"""TODO: add doc.

  Args:
    network_configs: A list of at least 1 `Tensor` objects with type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ConnectTPUEmbeddingHosts", name, network_configs)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_connect_tpu_embedding_hosts(
          (network_configs, name,), None)
      if _result is not NotImplemented:
        return _result
      return connect_tpu_embedding_hosts_eager_fallback(
          network_configs, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            connect_tpu_embedding_hosts, (), dict(network_configs=network_configs,
                                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_connect_tpu_embedding_hosts(
        (network_configs, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(network_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'network_configs' argument to "
        "'connect_tpu_embedding_hosts' Op, not %r." % network_configs)
  _attr_N = len(network_configs)
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ConnectTPUEmbeddingHosts", network_configs=network_configs,
                                    name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          connect_tpu_embedding_hosts, (), dict(network_configs=network_configs,
                                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ConnectTPUEmbeddingHosts = tf_export("raw_ops.ConnectTPUEmbeddingHosts")(_ops.to_raw_op(connect_tpu_embedding_hosts))
_dispatcher_for_connect_tpu_embedding_hosts = connect_tpu_embedding_hosts._tf_type_based_dispatcher.Dispatch


def connect_tpu_embedding_hosts_eager_fallback(network_configs, name, ctx):
  if not isinstance(network_configs, (list, tuple)):
    raise TypeError(
        "Expected list for 'network_configs' argument to "
        "'connect_tpu_embedding_hosts' Op, not %r." % network_configs)
  _attr_N = len(network_configs)
  network_configs = _ops.convert_n_to_tensor(network_configs, _dtypes.string)
  _inputs_flat = list(network_configs)
  _attrs = ("N", _attr_N)
  _result = _execute.execute(b"ConnectTPUEmbeddingHosts", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch')
def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal, combiners=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_indices_or_row_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: A `Tensor` of type `int32`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch", name,
        sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, device_ordinal, "combiners", combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch(
          (sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal, combiners,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(
          sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal,
          combiners=combiners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch, (), dict(sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                                           embedding_indices=embedding_indices,
                                                                           aggregation_weights=aggregation_weights,
                                                                           mode_override=mode_override,
                                                                           device_ordinal=device_ordinal,
                                                                           combiners=combiners,
                                                                           name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch(
        (sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, device_ordinal, combiners, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch", sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                          embedding_indices=embedding_indices,
                                                          aggregation_weights=aggregation_weights,
                                                          mode_override=mode_override,
                                                          device_ordinal=device_ordinal,
                                                          combiners=combiners,
                                                          name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch, (), dict(sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                                         embedding_indices=embedding_indices,
                                                                         aggregation_weights=aggregation_weights,
                                                                         mode_override=mode_override,
                                                                         device_ordinal=device_ordinal,
                                                                         combiners=combiners,
                                                                         name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
DynamicEnqueueTPUEmbeddingArbitraryTensorBatch = tf_export("raw_ops.DynamicEnqueueTPUEmbeddingArbitraryTensorBatch")(_ops.to_raw_op(dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch))
_dispatcher_for_dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch = dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch._tf_type_based_dispatcher.Dispatch


def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal, combiners, name, ctx):
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices_or_row_splits = _execute.args_to_matching_eager(list(sample_indices_or_row_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
  _inputs_flat = list(sample_indices_or_row_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override, device_ordinal]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "combiners", combiners)
  _result = _execute.execute(b"DynamicEnqueueTPUEmbeddingArbitraryTensorBatch",
                             0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_arbitrary_tensor_batch')
def enqueue_tpu_embedding_arbitrary_tensor_batch(sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal=-1, combiners=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_indices_or_row_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_indices_or_row_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingArbitraryTensorBatch", name,
        sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, "device_ordinal", device_ordinal, "combiners",
        combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_arbitrary_tensor_batch(
          (sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal, combiners,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(
          sample_indices_or_row_splits, embedding_indices,
          aggregation_weights, mode_override, device_ordinal=device_ordinal,
          combiners=combiners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_arbitrary_tensor_batch, (), dict(sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                                   embedding_indices=embedding_indices,
                                                                   aggregation_weights=aggregation_weights,
                                                                   mode_override=mode_override,
                                                                   device_ordinal=device_ordinal,
                                                                   combiners=combiners,
                                                                   name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_arbitrary_tensor_batch(
        (sample_indices_or_row_splits, embedding_indices, aggregation_weights,
        mode_override, device_ordinal, combiners, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingArbitraryTensorBatch", sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                   embedding_indices=embedding_indices,
                                                   aggregation_weights=aggregation_weights,
                                                   mode_override=mode_override,
                                                   device_ordinal=device_ordinal,
                                                   combiners=combiners,
                                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_arbitrary_tensor_batch, (), dict(sample_indices_or_row_splits=sample_indices_or_row_splits,
                                                                 embedding_indices=embedding_indices,
                                                                 aggregation_weights=aggregation_weights,
                                                                 mode_override=mode_override,
                                                                 device_ordinal=device_ordinal,
                                                                 combiners=combiners,
                                                                 name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingArbitraryTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingArbitraryTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_arbitrary_tensor_batch))
_dispatcher_for_enqueue_tpu_embedding_arbitrary_tensor_batch = enqueue_tpu_embedding_arbitrary_tensor_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits, embedding_indices, aggregation_weights, mode_override, device_ordinal, combiners, name, ctx):
  if not isinstance(sample_indices_or_row_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices_or_row_splits' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
  _attr_N = len(sample_indices_or_row_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices_or_row_splits'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices_or_row_splits = _execute.args_to_matching_eager(list(sample_indices_or_row_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices_or_row_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners)
  _result = _execute.execute(b"EnqueueTPUEmbeddingArbitraryTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_batch')
def enqueue_tpu_embedding_batch(batch, mode_override, device_ordinal=-1, combiners=[], name=None):
  r"""TODO: add doc.

  Args:
    batch: A list of at least 1 `Tensor` objects with type `string`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingBatch", name, batch, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_batch(
          (batch, mode_override, device_ordinal, combiners, name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_batch_eager_fallback(
          batch, mode_override, device_ordinal=device_ordinal,
          combiners=combiners, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_batch, (), dict(batch=batch,
                                                  mode_override=mode_override,
                                                  device_ordinal=device_ordinal,
                                                  combiners=combiners,
                                                  name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_batch(
        (batch, mode_override, device_ordinal, combiners, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingBatch", batch=batch, mode_override=mode_override,
                                    device_ordinal=device_ordinal,
                                    combiners=combiners, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_batch, (), dict(batch=batch,
                                                mode_override=mode_override,
                                                device_ordinal=device_ordinal,
                                                combiners=combiners,
                                                name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingBatch = tf_export("raw_ops.EnqueueTPUEmbeddingBatch")(_ops.to_raw_op(enqueue_tpu_embedding_batch))
_dispatcher_for_enqueue_tpu_embedding_batch = enqueue_tpu_embedding_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_batch_eager_fallback(batch, mode_override, device_ordinal, combiners, name, ctx):
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  batch = _ops.convert_n_to_tensor(batch, _dtypes.string)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(batch) + [mode_override]
  _attrs = ("N", _attr_N, "device_ordinal", device_ordinal, "combiners",
  combiners)
  _result = _execute.execute(b"EnqueueTPUEmbeddingBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_integer_batch')
def enqueue_tpu_embedding_integer_batch(batch, mode_override, device_ordinal=-1, name=None):
  r"""TODO: add doc.

  Args:
    batch: A list of at least 1 `Tensor` objects with type `int32`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingIntegerBatch", name, batch, mode_override,
        "device_ordinal", device_ordinal)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_integer_batch(
          (batch, mode_override, device_ordinal, name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_integer_batch_eager_fallback(
          batch, mode_override, device_ordinal=device_ordinal, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_integer_batch, (), dict(batch=batch,
                                                          mode_override=mode_override,
                                                          device_ordinal=device_ordinal,
                                                          name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_integer_batch(
        (batch, mode_override, device_ordinal, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_integer_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingIntegerBatch", batch=batch,
                                           mode_override=mode_override,
                                           device_ordinal=device_ordinal,
                                           name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_integer_batch, (), dict(batch=batch,
                                                        mode_override=mode_override,
                                                        device_ordinal=device_ordinal,
                                                        name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingIntegerBatch = tf_export("raw_ops.EnqueueTPUEmbeddingIntegerBatch")(_ops.to_raw_op(enqueue_tpu_embedding_integer_batch))
_dispatcher_for_enqueue_tpu_embedding_integer_batch = enqueue_tpu_embedding_integer_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_integer_batch_eager_fallback(batch, mode_override, device_ordinal, name, ctx):
  if not isinstance(batch, (list, tuple)):
    raise TypeError(
        "Expected list for 'batch' argument to "
        "'enqueue_tpu_embedding_integer_batch' Op, not %r." % batch)
  _attr_N = len(batch)
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  batch = _ops.convert_n_to_tensor(batch, _dtypes.int32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(batch) + [mode_override]
  _attrs = ("N", _attr_N, "device_ordinal", device_ordinal)
  _result = _execute.execute(b"EnqueueTPUEmbeddingIntegerBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_ragged_tensor_batch')
def enqueue_tpu_embedding_ragged_tensor_batch(sample_splits, embedding_indices, aggregation_weights, mode_override, table_ids, device_ordinal=-1, combiners=[], max_sequence_lengths=[], num_features=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    table_ids: A list of `ints`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    max_sequence_lengths: An optional list of `ints`. Defaults to `[]`.
    num_features: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingRaggedTensorBatch", name, sample_splits,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
        table_ids, "max_sequence_lengths", max_sequence_lengths,
        "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_ragged_tensor_batch(
          (sample_splits, embedding_indices, aggregation_weights,
          mode_override, table_ids, device_ordinal, combiners,
          max_sequence_lengths, num_features, name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(
          sample_splits, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          table_ids=table_ids, max_sequence_lengths=max_sequence_lengths,
          num_features=num_features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_ragged_tensor_batch, (), dict(sample_splits=sample_splits,
                                                                embedding_indices=embedding_indices,
                                                                aggregation_weights=aggregation_weights,
                                                                mode_override=mode_override,
                                                                table_ids=table_ids,
                                                                device_ordinal=device_ordinal,
                                                                combiners=combiners,
                                                                max_sequence_lengths=max_sequence_lengths,
                                                                num_features=num_features,
                                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_ragged_tensor_batch(
        (sample_splits, embedding_indices, aggregation_weights, mode_override,
        table_ids, device_ordinal, combiners, max_sequence_lengths,
        num_features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingRaggedTensorBatch", sample_splits=sample_splits,
                                                embedding_indices=embedding_indices,
                                                aggregation_weights=aggregation_weights,
                                                mode_override=mode_override,
                                                table_ids=table_ids,
                                                device_ordinal=device_ordinal,
                                                combiners=combiners,
                                                max_sequence_lengths=max_sequence_lengths,
                                                num_features=num_features,
                                                name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_ragged_tensor_batch, (), dict(sample_splits=sample_splits,
                                                              embedding_indices=embedding_indices,
                                                              aggregation_weights=aggregation_weights,
                                                              mode_override=mode_override,
                                                              table_ids=table_ids,
                                                              device_ordinal=device_ordinal,
                                                              combiners=combiners,
                                                              max_sequence_lengths=max_sequence_lengths,
                                                              num_features=num_features,
                                                              name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingRaggedTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingRaggedTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_ragged_tensor_batch))
_dispatcher_for_enqueue_tpu_embedding_ragged_tensor_batch = enqueue_tpu_embedding_ragged_tensor_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_ragged_tensor_batch_eager_fallback(sample_splits, embedding_indices, aggregation_weights, mode_override, table_ids, device_ordinal, combiners, max_sequence_lengths, num_features, name, ctx):
  if not isinstance(sample_splits, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_splits' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % sample_splits)
  _attr_N = len(sample_splits)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_ragged_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_splits'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_ragged_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _attr_T1, sample_splits = _execute.args_to_matching_eager(list(sample_splits), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
  table_ids, "max_sequence_lengths", max_sequence_lengths, "num_features",
  num_features)
  _result = _execute.execute(b"EnqueueTPUEmbeddingRaggedTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_sparse_batch')
def enqueue_tpu_embedding_sparse_batch(sample_indices, embedding_indices, aggregation_weights, mode_override, device_ordinal=-1, combiners=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_indices: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingSparseBatch", name, sample_indices,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_sparse_batch(
          (sample_indices, embedding_indices, aggregation_weights,
          mode_override, device_ordinal, combiners, name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_sparse_batch_eager_fallback(
          sample_indices, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_sparse_batch, (), dict(sample_indices=sample_indices,
                                                         embedding_indices=embedding_indices,
                                                         aggregation_weights=aggregation_weights,
                                                         mode_override=mode_override,
                                                         device_ordinal=device_ordinal,
                                                         combiners=combiners,
                                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_sparse_batch(
        (sample_indices, embedding_indices, aggregation_weights,
        mode_override, device_ordinal, combiners, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingSparseBatch", sample_indices=sample_indices,
                                          embedding_indices=embedding_indices,
                                          aggregation_weights=aggregation_weights,
                                          mode_override=mode_override,
                                          device_ordinal=device_ordinal,
                                          combiners=combiners, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_sparse_batch, (), dict(sample_indices=sample_indices,
                                                       embedding_indices=embedding_indices,
                                                       aggregation_weights=aggregation_weights,
                                                       mode_override=mode_override,
                                                       device_ordinal=device_ordinal,
                                                       combiners=combiners,
                                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingSparseBatch = tf_export("raw_ops.EnqueueTPUEmbeddingSparseBatch")(_ops.to_raw_op(enqueue_tpu_embedding_sparse_batch))
_dispatcher_for_enqueue_tpu_embedding_sparse_batch = enqueue_tpu_embedding_sparse_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_sparse_batch_eager_fallback(sample_indices, embedding_indices, aggregation_weights, mode_override, device_ordinal, combiners, name, ctx):
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  _attr_T1, sample_indices = _execute.args_to_matching_eager(list(sample_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners)
  _result = _execute.execute(b"EnqueueTPUEmbeddingSparseBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('enqueue_tpu_embedding_sparse_tensor_batch')
def enqueue_tpu_embedding_sparse_tensor_batch(sample_indices, embedding_indices, aggregation_weights, mode_override, table_ids, device_ordinal=-1, combiners=[], max_sequence_lengths=[], num_features=[], name=None):
  r"""TODO: add doc.

  Args:
    sample_indices: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
    embedding_indices: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `int32`, `int64`.
    aggregation_weights: A list with the same length as `sample_indices` of `Tensor` objects with the same type in: `float32`, `float64`.
    mode_override: A `Tensor` of type `string`.
    table_ids: A list of `ints`.
    device_ordinal: An optional `int`. Defaults to `-1`.
    combiners: An optional list of `strings`. Defaults to `[]`.
    max_sequence_lengths: An optional list of `ints`. Defaults to `[]`.
    num_features: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "EnqueueTPUEmbeddingSparseTensorBatch", name, sample_indices,
        embedding_indices, aggregation_weights, mode_override,
        "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
        table_ids, "max_sequence_lengths", max_sequence_lengths,
        "num_features", num_features)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_enqueue_tpu_embedding_sparse_tensor_batch(
          (sample_indices, embedding_indices, aggregation_weights,
          mode_override, table_ids, device_ordinal, combiners,
          max_sequence_lengths, num_features, name,), None)
      if _result is not NotImplemented:
        return _result
      return enqueue_tpu_embedding_sparse_tensor_batch_eager_fallback(
          sample_indices, embedding_indices, aggregation_weights,
          mode_override, device_ordinal=device_ordinal, combiners=combiners,
          table_ids=table_ids, max_sequence_lengths=max_sequence_lengths,
          num_features=num_features, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            enqueue_tpu_embedding_sparse_tensor_batch, (), dict(sample_indices=sample_indices,
                                                                embedding_indices=embedding_indices,
                                                                aggregation_weights=aggregation_weights,
                                                                mode_override=mode_override,
                                                                table_ids=table_ids,
                                                                device_ordinal=device_ordinal,
                                                                combiners=combiners,
                                                                max_sequence_lengths=max_sequence_lengths,
                                                                num_features=num_features,
                                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_enqueue_tpu_embedding_sparse_tensor_batch(
        (sample_indices, embedding_indices, aggregation_weights,
        mode_override, table_ids, device_ordinal, combiners,
        max_sequence_lengths, num_features, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "EnqueueTPUEmbeddingSparseTensorBatch", sample_indices=sample_indices,
                                                embedding_indices=embedding_indices,
                                                aggregation_weights=aggregation_weights,
                                                mode_override=mode_override,
                                                table_ids=table_ids,
                                                device_ordinal=device_ordinal,
                                                combiners=combiners,
                                                max_sequence_lengths=max_sequence_lengths,
                                                num_features=num_features,
                                                name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          enqueue_tpu_embedding_sparse_tensor_batch, (), dict(sample_indices=sample_indices,
                                                              embedding_indices=embedding_indices,
                                                              aggregation_weights=aggregation_weights,
                                                              mode_override=mode_override,
                                                              table_ids=table_ids,
                                                              device_ordinal=device_ordinal,
                                                              combiners=combiners,
                                                              max_sequence_lengths=max_sequence_lengths,
                                                              num_features=num_features,
                                                              name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
EnqueueTPUEmbeddingSparseTensorBatch = tf_export("raw_ops.EnqueueTPUEmbeddingSparseTensorBatch")(_ops.to_raw_op(enqueue_tpu_embedding_sparse_tensor_batch))
_dispatcher_for_enqueue_tpu_embedding_sparse_tensor_batch = enqueue_tpu_embedding_sparse_tensor_batch._tf_type_based_dispatcher.Dispatch


def enqueue_tpu_embedding_sparse_tensor_batch_eager_fallback(sample_indices, embedding_indices, aggregation_weights, mode_override, table_ids, device_ordinal, combiners, max_sequence_lengths, num_features, name, ctx):
  if not isinstance(sample_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'sample_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % sample_indices)
  _attr_N = len(sample_indices)
  if not isinstance(embedding_indices, (list, tuple)):
    raise TypeError(
        "Expected list for 'embedding_indices' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % embedding_indices)
  if len(embedding_indices) != _attr_N:
    raise ValueError(
        "List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(embedding_indices), _attr_N))
  if not isinstance(aggregation_weights, (list, tuple)):
    raise TypeError(
        "Expected list for 'aggregation_weights' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % aggregation_weights)
  if len(aggregation_weights) != _attr_N:
    raise ValueError(
        "List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_tensor_batch' Op with length %d "
        "must match length %d of argument 'sample_indices'." %
        (len(aggregation_weights), _attr_N))
  if not isinstance(table_ids, (list, tuple)):
    raise TypeError(
        "Expected list for 'table_ids' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % table_ids)
  table_ids = [_execute.make_int(_i, "table_ids") for _i in table_ids]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  if combiners is None:
    combiners = []
  if not isinstance(combiners, (list, tuple)):
    raise TypeError(
        "Expected list for 'combiners' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % combiners)
  combiners = [_execute.make_str(_s, "combiners") for _s in combiners]
  if max_sequence_lengths is None:
    max_sequence_lengths = []
  if not isinstance(max_sequence_lengths, (list, tuple)):
    raise TypeError(
        "Expected list for 'max_sequence_lengths' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % max_sequence_lengths)
  max_sequence_lengths = [_execute.make_int(_i, "max_sequence_lengths") for _i in max_sequence_lengths]
  if num_features is None:
    num_features = []
  if not isinstance(num_features, (list, tuple)):
    raise TypeError(
        "Expected list for 'num_features' argument to "
        "'enqueue_tpu_embedding_sparse_tensor_batch' Op, not %r." % num_features)
  num_features = [_execute.make_int(_i, "num_features") for _i in num_features]
  _attr_T1, sample_indices = _execute.args_to_matching_eager(list(sample_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64, ], _dtypes.int32)
  _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64, ], _dtypes.float32)
  mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
  _inputs_flat = list(sample_indices) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
  _attrs = ("T1", _attr_T1, "T2", _attr_T2, "T3", _attr_T3, "N", _attr_N,
  "device_ordinal", device_ordinal, "combiners", combiners, "table_ids",
  table_ids, "max_sequence_lengths", max_sequence_lengths, "num_features",
  num_features)
  _result = _execute.execute(b"EnqueueTPUEmbeddingSparseTensorBatch", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('execute_tpu_embedding_partitioner')
def execute_tpu_embedding_partitioner(config, name=None):
  r"""TODO: add doc.

  Args:
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ExecuteTPUEmbeddingPartitioner", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_execute_tpu_embedding_partitioner(
          (config, name,), None)
      if _result is not NotImplemented:
        return _result
      return execute_tpu_embedding_partitioner_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            execute_tpu_embedding_partitioner, (), dict(config=config,
                                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_execute_tpu_embedding_partitioner(
        (config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ExecuteTPUEmbeddingPartitioner", config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          execute_tpu_embedding_partitioner, (), dict(config=config,
                                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "ExecuteTPUEmbeddingPartitioner", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

ExecuteTPUEmbeddingPartitioner = tf_export("raw_ops.ExecuteTPUEmbeddingPartitioner")(_ops.to_raw_op(execute_tpu_embedding_partitioner))
_dispatcher_for_execute_tpu_embedding_partitioner = execute_tpu_embedding_partitioner._tf_type_based_dispatcher.Dispatch


def execute_tpu_embedding_partitioner_eager_fallback(config, name, ctx):
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"ExecuteTPUEmbeddingPartitioner", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "ExecuteTPUEmbeddingPartitioner", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('finalize_tpu_embedding')
def finalize_tpu_embedding(common_config, memory_config, name=None):
  r"""TODO: add doc.

  Args:
    common_config: A `Tensor` of type `string`.
    memory_config: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "FinalizeTPUEmbedding", name, common_config, memory_config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_finalize_tpu_embedding(
          (common_config, memory_config, name,), None)
      if _result is not NotImplemented:
        return _result
      return finalize_tpu_embedding_eager_fallback(
          common_config, memory_config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            finalize_tpu_embedding, (), dict(common_config=common_config,
                                             memory_config=memory_config,
                                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_finalize_tpu_embedding(
        (common_config, memory_config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FinalizeTPUEmbedding", common_config=common_config,
                                memory_config=memory_config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          finalize_tpu_embedding, (), dict(common_config=common_config,
                                           memory_config=memory_config,
                                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
FinalizeTPUEmbedding = tf_export("raw_ops.FinalizeTPUEmbedding")(_ops.to_raw_op(finalize_tpu_embedding))
_dispatcher_for_finalize_tpu_embedding = finalize_tpu_embedding._tf_type_based_dispatcher.Dispatch


def finalize_tpu_embedding_eager_fallback(common_config, memory_config, name, ctx):
  common_config = _ops.convert_to_tensor(common_config, _dtypes.string)
  memory_config = _ops.convert_to_tensor(memory_config, _dtypes.string)
  _inputs_flat = [common_config, memory_config]
  _attrs = None
  _result = _execute.execute(b"FinalizeTPUEmbedding", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('is_tpu_embedding_initialized')
def is_tpu_embedding_initialized(config="", name=None):
  r"""TODO: add doc.

  Args:
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "IsTPUEmbeddingInitialized", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_is_tpu_embedding_initialized(
          (config, name,), None)
      if _result is not NotImplemented:
        return _result
      return is_tpu_embedding_initialized_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            is_tpu_embedding_initialized, (), dict(config=config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_is_tpu_embedding_initialized(
        (config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "IsTPUEmbeddingInitialized", config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          is_tpu_embedding_initialized, (), dict(config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "IsTPUEmbeddingInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

IsTPUEmbeddingInitialized = tf_export("raw_ops.IsTPUEmbeddingInitialized")(_ops.to_raw_op(is_tpu_embedding_initialized))
_dispatcher_for_is_tpu_embedding_initialized = is_tpu_embedding_initialized._tf_type_based_dispatcher.Dispatch


def is_tpu_embedding_initialized_eager_fallback(config, name, ctx):
  if config is None:
    config = ""
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"IsTPUEmbeddingInitialized", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "IsTPUEmbeddingInitialized", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('load_all_tpu_embedding_parameters')
def load_all_tpu_embedding_parameters(parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4, auxiliary5, auxiliary6, auxiliary7, config, num_shards, shard_id, name=None):
  r"""TODO: add doc.

  Args:
    parameters: A list of at least 1 `Tensor` objects with type `float32`.
    auxiliary1: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary2: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary3: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary4: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary5: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary6: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    auxiliary7: A list with the same length as `parameters` of `Tensor` objects with type `float32`.
    config: A `string`.
    num_shards: An `int`.
    shard_id: An `int`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "LoadAllTPUEmbeddingParameters", name, parameters, auxiliary1,
        auxiliary2, auxiliary3, auxiliary4, auxiliary5, auxiliary6,
        auxiliary7, "config", config, "num_shards", num_shards, "shard_id",
        shard_id)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_load_all_tpu_embedding_parameters(
          (parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4,
          auxiliary5, auxiliary6, auxiliary7, config, num_shards, shard_id,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return load_all_tpu_embedding_parameters_eager_fallback(
          parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4,
          auxiliary5, auxiliary6, auxiliary7, config=config,
          num_shards=num_shards, shard_id=shard_id, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            load_all_tpu_embedding_parameters, (), dict(parameters=parameters,
                                                        auxiliary1=auxiliary1,
                                                        auxiliary2=auxiliary2,
                                                        auxiliary3=auxiliary3,
                                                        auxiliary4=auxiliary4,
                                                        auxiliary5=auxiliary5,
                                                        auxiliary6=auxiliary6,
                                                        auxiliary7=auxiliary7,
                                                        config=config,
                                                        num_shards=num_shards,
                                                        shard_id=shard_id,
                                                        name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_load_all_tpu_embedding_parameters(
        (parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4,
        auxiliary5, auxiliary6, auxiliary7, config, num_shards, shard_id,
        name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(parameters, (list, tuple)):
    raise TypeError(
        "Expected list for 'parameters' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % parameters)
  _attr_NumTables = len(parameters)
  if not isinstance(auxiliary1, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary1' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary1)
  if len(auxiliary1) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary1' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary1), _attr_NumTables))
  if not isinstance(auxiliary2, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary2' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary2)
  if len(auxiliary2) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary2' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary2), _attr_NumTables))
  if not isinstance(auxiliary3, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary3' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary3)
  if len(auxiliary3) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary3' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary3), _attr_NumTables))
  if not isinstance(auxiliary4, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary4' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary4)
  if len(auxiliary4) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary4' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary4), _attr_NumTables))
  if not isinstance(auxiliary5, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary5' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary5)
  if len(auxiliary5) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary5' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary5), _attr_NumTables))
  if not isinstance(auxiliary6, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary6' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary6)
  if len(auxiliary6) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary6' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary6), _attr_NumTables))
  if not isinstance(auxiliary7, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary7' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary7)
  if len(auxiliary7) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary7' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary7), _attr_NumTables))
  config = _execute.make_str(config, "config")
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "LoadAllTPUEmbeddingParameters", parameters=parameters,
                                         auxiliary1=auxiliary1,
                                         auxiliary2=auxiliary2,
                                         auxiliary3=auxiliary3,
                                         auxiliary4=auxiliary4,
                                         auxiliary5=auxiliary5,
                                         auxiliary6=auxiliary6,
                                         auxiliary7=auxiliary7, config=config,
                                         num_shards=num_shards,
                                         shard_id=shard_id, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          load_all_tpu_embedding_parameters, (), dict(parameters=parameters,
                                                      auxiliary1=auxiliary1,
                                                      auxiliary2=auxiliary2,
                                                      auxiliary3=auxiliary3,
                                                      auxiliary4=auxiliary4,
                                                      auxiliary5=auxiliary5,
                                                      auxiliary6=auxiliary6,
                                                      auxiliary7=auxiliary7,
                                                      config=config,
                                                      num_shards=num_shards,
                                                      shard_id=shard_id,
                                                      name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
LoadAllTPUEmbeddingParameters = tf_export("raw_ops.LoadAllTPUEmbeddingParameters")(_ops.to_raw_op(load_all_tpu_embedding_parameters))
_dispatcher_for_load_all_tpu_embedding_parameters = load_all_tpu_embedding_parameters._tf_type_based_dispatcher.Dispatch


def load_all_tpu_embedding_parameters_eager_fallback(parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4, auxiliary5, auxiliary6, auxiliary7, config, num_shards, shard_id, name, ctx):
  if not isinstance(parameters, (list, tuple)):
    raise TypeError(
        "Expected list for 'parameters' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % parameters)
  _attr_NumTables = len(parameters)
  if not isinstance(auxiliary1, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary1' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary1)
  if len(auxiliary1) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary1' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary1), _attr_NumTables))
  if not isinstance(auxiliary2, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary2' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary2)
  if len(auxiliary2) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary2' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary2), _attr_NumTables))
  if not isinstance(auxiliary3, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary3' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary3)
  if len(auxiliary3) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary3' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary3), _attr_NumTables))
  if not isinstance(auxiliary4, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary4' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary4)
  if len(auxiliary4) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary4' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary4), _attr_NumTables))
  if not isinstance(auxiliary5, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary5' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary5)
  if len(auxiliary5) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary5' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary5), _attr_NumTables))
  if not isinstance(auxiliary6, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary6' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary6)
  if len(auxiliary6) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary6' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary6), _attr_NumTables))
  if not isinstance(auxiliary7, (list, tuple)):
    raise TypeError(
        "Expected list for 'auxiliary7' argument to "
        "'load_all_tpu_embedding_parameters' Op, not %r." % auxiliary7)
  if len(auxiliary7) != _attr_NumTables:
    raise ValueError(
        "List argument 'auxiliary7' to 'load_all_tpu_embedding_parameters' Op with length %d "
        "must match length %d of argument 'parameters'." %
        (len(auxiliary7), _attr_NumTables))
  config = _execute.make_str(config, "config")
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  parameters = _ops.convert_n_to_tensor(parameters, _dtypes.float32)
  auxiliary1 = _ops.convert_n_to_tensor(auxiliary1, _dtypes.float32)
  auxiliary2 = _ops.convert_n_to_tensor(auxiliary2, _dtypes.float32)
  auxiliary3 = _ops.convert_n_to_tensor(auxiliary3, _dtypes.float32)
  auxiliary4 = _ops.convert_n_to_tensor(auxiliary4, _dtypes.float32)
  auxiliary5 = _ops.convert_n_to_tensor(auxiliary5, _dtypes.float32)
  auxiliary6 = _ops.convert_n_to_tensor(auxiliary6, _dtypes.float32)
  auxiliary7 = _ops.convert_n_to_tensor(auxiliary7, _dtypes.float32)
  _inputs_flat = list(parameters) + list(auxiliary1) + list(auxiliary2) + list(auxiliary3) + list(auxiliary4) + list(auxiliary5) + list(auxiliary6) + list(auxiliary7)
  _attrs = ("NumTables", _attr_NumTables, "config", config, "num_shards",
  num_shards, "shard_id", shard_id)
  _result = _execute.execute(b"LoadAllTPUEmbeddingParameters", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('recv_tpu_embedding_activations')
def recv_tpu_embedding_activations(num_outputs, config, name=None):
  r"""TODO: add doc.

  Args:
    num_outputs: An `int` that is `>= 1`.
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_outputs` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RecvTPUEmbeddingActivations", name, "num_outputs", num_outputs,
        "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_recv_tpu_embedding_activations(
          (num_outputs, config, name,), None)
      if _result is not NotImplemented:
        return _result
      return recv_tpu_embedding_activations_eager_fallback(
          num_outputs=num_outputs, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            recv_tpu_embedding_activations, (), dict(num_outputs=num_outputs,
                                                     config=config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_recv_tpu_embedding_activations(
        (num_outputs, config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  num_outputs = _execute.make_int(num_outputs, "num_outputs")
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RecvTPUEmbeddingActivations", num_outputs=num_outputs, config=config,
                                       name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          recv_tpu_embedding_activations, (), dict(num_outputs=num_outputs,
                                                   config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("num_outputs", _op._get_attr_int("num_outputs"), "config",
              _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result

RecvTPUEmbeddingActivations = tf_export("raw_ops.RecvTPUEmbeddingActivations")(_ops.to_raw_op(recv_tpu_embedding_activations))
_dispatcher_for_recv_tpu_embedding_activations = recv_tpu_embedding_activations._tf_type_based_dispatcher.Dispatch


def recv_tpu_embedding_activations_eager_fallback(num_outputs, config, name, ctx):
  num_outputs = _execute.make_int(num_outputs, "num_outputs")
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("num_outputs", num_outputs, "config", config)
  _result = _execute.execute(b"RecvTPUEmbeddingActivations", num_outputs,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result

_RetrieveAllTPUEmbeddingParametersOutput = collections.namedtuple(
    "RetrieveAllTPUEmbeddingParameters",
    ["parameters", "auxiliary1", "auxiliary2", "auxiliary3", "auxiliary4", "auxiliary5", "auxiliary6", "auxiliary7"])


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('retrieve_all_tpu_embedding_parameters')
def retrieve_all_tpu_embedding_parameters(NumTables, config, num_shards, shard_id, name=None):
  r"""TODO: add doc.

  Args:
    NumTables: An `int` that is `>= 1`.
    config: A `string`.
    num_shards: An `int`.
    shard_id: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, auxiliary1, auxiliary2, auxiliary3, auxiliary4, auxiliary5, auxiliary6, auxiliary7).

    parameters: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary1: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary2: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary3: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary4: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary5: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary6: A list of `NumTables` `Tensor` objects with type `float32`.
    auxiliary7: A list of `NumTables` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "RetrieveAllTPUEmbeddingParameters", name, "NumTables",
        NumTables, "config", config, "num_shards", num_shards, "shard_id",
        shard_id)
      _result = _RetrieveAllTPUEmbeddingParametersOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_retrieve_all_tpu_embedding_parameters(
          (NumTables, config, num_shards, shard_id, name,), None)
      if _result is not NotImplemented:
        return _result
      return retrieve_all_tpu_embedding_parameters_eager_fallback(
          NumTables=NumTables, config=config, num_shards=num_shards,
          shard_id=shard_id, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            retrieve_all_tpu_embedding_parameters, (), dict(NumTables=NumTables,
                                                            config=config,
                                                            num_shards=num_shards,
                                                            shard_id=shard_id,
                                                            name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_retrieve_all_tpu_embedding_parameters(
        (NumTables, config, num_shards, shard_id, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  NumTables = _execute.make_int(NumTables, "NumTables")
  config = _execute.make_str(config, "config")
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "RetrieveAllTPUEmbeddingParameters", NumTables=NumTables,
                                             config=config,
                                             num_shards=num_shards,
                                             shard_id=shard_id, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          retrieve_all_tpu_embedding_parameters, (), dict(NumTables=NumTables,
                                                          config=config,
                                                          num_shards=num_shards,
                                                          shard_id=shard_id,
                                                          name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("NumTables", _op._get_attr_int("NumTables"), "config",
              _op.get_attr("config"), "num_shards",
              _op._get_attr_int("num_shards"), "shard_id",
              _op._get_attr_int("shard_id"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "RetrieveAllTPUEmbeddingParameters", _inputs_flat, _attrs, _result)
  _result = [_result[:NumTables]] + _result[NumTables:]
  _result = _result[:1] + [_result[1:1 + NumTables]] + _result[1 + NumTables:]
  _result = _result[:2] + [_result[2:2 + NumTables]] + _result[2 + NumTables:]
  _result = _result[:3] + [_result[3:3 + NumTables]] + _result[3 + NumTables:]
  _result = _result[:4] + [_result[4:4 + NumTables]] + _result[4 + NumTables:]
  _result = _result[:5] + [_result[5:5 + NumTables]] + _result[5 + NumTables:]
  _result = _result[:6] + [_result[6:6 + NumTables]] + _result[6 + NumTables:]
  _result = _result[:7] + [_result[7:]]
  _result = _RetrieveAllTPUEmbeddingParametersOutput._make(_result)
  return _result

RetrieveAllTPUEmbeddingParameters = tf_export("raw_ops.RetrieveAllTPUEmbeddingParameters")(_ops.to_raw_op(retrieve_all_tpu_embedding_parameters))
_dispatcher_for_retrieve_all_tpu_embedding_parameters = retrieve_all_tpu_embedding_parameters._tf_type_based_dispatcher.Dispatch


def retrieve_all_tpu_embedding_parameters_eager_fallback(NumTables, config, num_shards, shard_id, name, ctx):
  NumTables = _execute.make_int(NumTables, "NumTables")
  config = _execute.make_str(config, "config")
  num_shards = _execute.make_int(num_shards, "num_shards")
  shard_id = _execute.make_int(shard_id, "shard_id")
  _inputs_flat = []
  _attrs = ("NumTables", NumTables, "config", config, "num_shards",
  num_shards, "shard_id", shard_id)
  _result = _execute.execute(b"RetrieveAllTPUEmbeddingParameters", NumTables +
                             NumTables + NumTables + NumTables + NumTables +
                             NumTables + NumTables + NumTables,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "RetrieveAllTPUEmbeddingParameters", _inputs_flat, _attrs, _result)
  _result = [_result[:NumTables]] + _result[NumTables:]
  _result = _result[:1] + [_result[1:1 + NumTables]] + _result[1 + NumTables:]
  _result = _result[:2] + [_result[2:2 + NumTables]] + _result[2 + NumTables:]
  _result = _result[:3] + [_result[3:3 + NumTables]] + _result[3 + NumTables:]
  _result = _result[:4] + [_result[4:4 + NumTables]] + _result[4 + NumTables:]
  _result = _result[:5] + [_result[5:5 + NumTables]] + _result[5 + NumTables:]
  _result = _result[:6] + [_result[6:6 + NumTables]] + _result[6 + NumTables:]
  _result = _result[:7] + [_result[7:]]
  _result = _RetrieveAllTPUEmbeddingParametersOutput._make(_result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('send_tpu_embedding_gradients')
def send_tpu_embedding_gradients(inputs, learning_rates, config, name=None):
  r"""TODO: add doc.

  Args:
    inputs: A list of at least 1 `Tensor` objects with type `float32`.
    learning_rates: A list of `Tensor` objects with type `float32`.
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SendTPUEmbeddingGradients", name, inputs, learning_rates,
        "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_send_tpu_embedding_gradients(
          (inputs, learning_rates, config, name,), None)
      if _result is not NotImplemented:
        return _result
      return send_tpu_embedding_gradients_eager_fallback(
          inputs, learning_rates, config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            send_tpu_embedding_gradients, (), dict(inputs=inputs,
                                                   learning_rates=learning_rates,
                                                   config=config, name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_send_tpu_embedding_gradients(
        (inputs, learning_rates, config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NN = len(learning_rates)
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SendTPUEmbeddingGradients", inputs=inputs,
                                     learning_rates=learning_rates,
                                     config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          send_tpu_embedding_gradients, (), dict(inputs=inputs,
                                                 learning_rates=learning_rates,
                                                 config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
SendTPUEmbeddingGradients = tf_export("raw_ops.SendTPUEmbeddingGradients")(_ops.to_raw_op(send_tpu_embedding_gradients))
_dispatcher_for_send_tpu_embedding_gradients = send_tpu_embedding_gradients._tf_type_based_dispatcher.Dispatch


def send_tpu_embedding_gradients_eager_fallback(inputs, learning_rates, config, name, ctx):
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NN = len(learning_rates)
  config = _execute.make_str(config, "config")
  inputs = _ops.convert_n_to_tensor(inputs, _dtypes.float32)
  learning_rates = _ops.convert_n_to_tensor(learning_rates, _dtypes.float32)
  _inputs_flat = list(inputs) + list(learning_rates)
  _attrs = ("N", _attr_N, "NN", _attr_NN, "config", config)
  _result = _execute.execute(b"SendTPUEmbeddingGradients", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('shutdown_distributed_tpu')
def shutdown_distributed_tpu(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "ShutdownDistributedTPU", name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_shutdown_distributed_tpu(
          (name,), None)
      if _result is not NotImplemented:
        return _result
      return shutdown_distributed_tpu_eager_fallback(
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            shutdown_distributed_tpu, (), dict(name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_shutdown_distributed_tpu(
        (name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ShutdownDistributedTPU", name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          shutdown_distributed_tpu, (), dict(name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
ShutdownDistributedTPU = tf_export("raw_ops.ShutdownDistributedTPU")(_ops.to_raw_op(shutdown_distributed_tpu))
_dispatcher_for_shutdown_distributed_tpu = shutdown_distributed_tpu._tf_type_based_dispatcher.Dispatch


def shutdown_distributed_tpu_eager_fallback(name, ctx):
  _inputs_flat = []
  _attrs = None
  _result = _execute.execute(b"ShutdownDistributedTPU", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('tpu_embedding_activations')
def tpu_embedding_activations(embedding_variable, sliced_activations, table_id, lookup_id, name=None):
  r"""TODO: add doc.

  Args:
    embedding_variable: A `Tensor` of type `float32`.
    sliced_activations: A `Tensor` of type `float32`.
    table_id: An `int` that is `>= 0`.
    lookup_id: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "TPUEmbeddingActivations", name, embedding_variable,
        sliced_activations, "table_id", table_id, "lookup_id", lookup_id)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_tpu_embedding_activations(
          (embedding_variable, sliced_activations, table_id, lookup_id,
          name,), None)
      if _result is not NotImplemented:
        return _result
      return tpu_embedding_activations_eager_fallback(
          embedding_variable, sliced_activations, table_id=table_id,
          lookup_id=lookup_id, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            tpu_embedding_activations, (), dict(embedding_variable=embedding_variable,
                                                sliced_activations=sliced_activations,
                                                table_id=table_id,
                                                lookup_id=lookup_id,
                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_tpu_embedding_activations(
        (embedding_variable, sliced_activations, table_id, lookup_id, name,),
        None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  table_id = _execute.make_int(table_id, "table_id")
  lookup_id = _execute.make_int(lookup_id, "lookup_id")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "TPUEmbeddingActivations", embedding_variable=embedding_variable,
                                   sliced_activations=sliced_activations,
                                   table_id=table_id, lookup_id=lookup_id,
                                   name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          tpu_embedding_activations, (), dict(embedding_variable=embedding_variable,
                                              sliced_activations=sliced_activations,
                                              table_id=table_id,
                                              lookup_id=lookup_id, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("table_id", _op._get_attr_int("table_id"), "lookup_id",
              _op._get_attr_int("lookup_id"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "TPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

TPUEmbeddingActivations = tf_export("raw_ops.TPUEmbeddingActivations")(_ops.to_raw_op(tpu_embedding_activations))
_dispatcher_for_tpu_embedding_activations = tpu_embedding_activations._tf_type_based_dispatcher.Dispatch


def tpu_embedding_activations_eager_fallback(embedding_variable, sliced_activations, table_id, lookup_id, name, ctx):
  table_id = _execute.make_int(table_id, "table_id")
  lookup_id = _execute.make_int(lookup_id, "lookup_id")
  embedding_variable = _ops.convert_to_tensor(embedding_variable, _dtypes.float32)
  sliced_activations = _ops.convert_to_tensor(sliced_activations, _dtypes.float32)
  _inputs_flat = [embedding_variable, sliced_activations]
  _attrs = ("table_id", table_id, "lookup_id", lookup_id)
  _result = _execute.execute(b"TPUEmbeddingActivations", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "TPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_recv_tpu_embedding_activations')
def xla_recv_tpu_embedding_activations(deduplication_data, num_tables, config, name=None):
  r"""TODO: add doc.

  Args:
    deduplication_data: A `Tensor` of type `variant`.
    num_tables: An `int` that is `>= 1`.
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_tables` `Tensor` objects with type `float32`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRecvTPUEmbeddingActivations", name, deduplication_data,
        "num_tables", num_tables, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_recv_tpu_embedding_activations(
          (deduplication_data, num_tables, config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_recv_tpu_embedding_activations_eager_fallback(
          deduplication_data, num_tables=num_tables, config=config, name=name,
          ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_recv_tpu_embedding_activations, (), dict(deduplication_data=deduplication_data,
                                                         num_tables=num_tables,
                                                         config=config,
                                                         name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_recv_tpu_embedding_activations(
        (deduplication_data, num_tables, config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  num_tables = _execute.make_int(num_tables, "num_tables")
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRecvTPUEmbeddingActivations", deduplication_data=deduplication_data,
                                          num_tables=num_tables,
                                          config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_recv_tpu_embedding_activations, (), dict(deduplication_data=deduplication_data,
                                                       num_tables=num_tables,
                                                       config=config,
                                                       name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if not _result:
    return _op
  if _execute.must_record_gradient():
    _attrs = ("num_tables", _op._get_attr_int("num_tables"), "config",
              _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result

XlaRecvTPUEmbeddingActivations = tf_export("raw_ops.XlaRecvTPUEmbeddingActivations")(_ops.to_raw_op(xla_recv_tpu_embedding_activations))
_dispatcher_for_xla_recv_tpu_embedding_activations = xla_recv_tpu_embedding_activations._tf_type_based_dispatcher.Dispatch


def xla_recv_tpu_embedding_activations_eager_fallback(deduplication_data, num_tables, config, name, ctx):
  num_tables = _execute.make_int(num_tables, "num_tables")
  config = _execute.make_str(config, "config")
  deduplication_data = _ops.convert_to_tensor(deduplication_data, _dtypes.variant)
  _inputs_flat = [deduplication_data]
  _attrs = ("num_tables", num_tables, "config", config)
  _result = _execute.execute(b"XlaRecvTPUEmbeddingActivations", num_tables,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRecvTPUEmbeddingActivations", _inputs_flat, _attrs, _result)
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_recv_tpu_embedding_deduplication_data')
def xla_recv_tpu_embedding_deduplication_data(config, name=None):
  r"""TODO: add doc.

  Args:
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaRecvTPUEmbeddingDeduplicationData", name, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_recv_tpu_embedding_deduplication_data(
          (config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_recv_tpu_embedding_deduplication_data_eager_fallback(
          config=config, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_recv_tpu_embedding_deduplication_data, (), dict(config=config,
                                                                name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_recv_tpu_embedding_deduplication_data(
        (config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaRecvTPUEmbeddingDeduplicationData", config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_recv_tpu_embedding_deduplication_data, (), dict(config=config,
                                                              name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("config", _op.get_attr("config"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "XlaRecvTPUEmbeddingDeduplicationData", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

XlaRecvTPUEmbeddingDeduplicationData = tf_export("raw_ops.XlaRecvTPUEmbeddingDeduplicationData")(_ops.to_raw_op(xla_recv_tpu_embedding_deduplication_data))
_dispatcher_for_xla_recv_tpu_embedding_deduplication_data = xla_recv_tpu_embedding_deduplication_data._tf_type_based_dispatcher.Dispatch


def xla_recv_tpu_embedding_deduplication_data_eager_fallback(config, name, ctx):
  config = _execute.make_str(config, "config")
  _inputs_flat = []
  _attrs = ("config", config)
  _result = _execute.execute(b"XlaRecvTPUEmbeddingDeduplicationData", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "XlaRecvTPUEmbeddingDeduplicationData", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_send_tpu_embedding_gradients')
def xla_send_tpu_embedding_gradients(gradients, learning_rates, deduplication_data, config, name=None):
  r"""TODO: add doc.

  Args:
    gradients: A list of at least 1 `Tensor` objects with type `float32`.
    learning_rates: A list of `Tensor` objects with type `float32`.
    deduplication_data: A `Tensor` of type `variant`.
    config: A `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "XlaSendTPUEmbeddingGradients", name, gradients, learning_rates,
        deduplication_data, "config", config)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_xla_send_tpu_embedding_gradients(
          (gradients, learning_rates, deduplication_data, config, name,), None)
      if _result is not NotImplemented:
        return _result
      return xla_send_tpu_embedding_gradients_eager_fallback(
          gradients, learning_rates, deduplication_data, config=config,
          name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            xla_send_tpu_embedding_gradients, (), dict(gradients=gradients,
                                                       learning_rates=learning_rates,
                                                       deduplication_data=deduplication_data,
                                                       config=config,
                                                       name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_xla_send_tpu_embedding_gradients(
        (gradients, learning_rates, deduplication_data, config, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  if not isinstance(gradients, (list, tuple)):
    raise TypeError(
        "Expected list for 'gradients' argument to "
        "'xla_send_tpu_embedding_gradients' Op, not %r." % gradients)
  _attr_NumTables = len(gradients)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'xla_send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NumLearningRateTags = len(learning_rates)
  config = _execute.make_str(config, "config")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "XlaSendTPUEmbeddingGradients", gradients=gradients,
                                        learning_rates=learning_rates,
                                        deduplication_data=deduplication_data,
                                        config=config, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          xla_send_tpu_embedding_gradients, (), dict(gradients=gradients,
                                                     learning_rates=learning_rates,
                                                     deduplication_data=deduplication_data,
                                                     config=config, name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  return _op
XlaSendTPUEmbeddingGradients = tf_export("raw_ops.XlaSendTPUEmbeddingGradients")(_ops.to_raw_op(xla_send_tpu_embedding_gradients))
_dispatcher_for_xla_send_tpu_embedding_gradients = xla_send_tpu_embedding_gradients._tf_type_based_dispatcher.Dispatch


def xla_send_tpu_embedding_gradients_eager_fallback(gradients, learning_rates, deduplication_data, config, name, ctx):
  if not isinstance(gradients, (list, tuple)):
    raise TypeError(
        "Expected list for 'gradients' argument to "
        "'xla_send_tpu_embedding_gradients' Op, not %r." % gradients)
  _attr_NumTables = len(gradients)
  if not isinstance(learning_rates, (list, tuple)):
    raise TypeError(
        "Expected list for 'learning_rates' argument to "
        "'xla_send_tpu_embedding_gradients' Op, not %r." % learning_rates)
  _attr_NumLearningRateTags = len(learning_rates)
  config = _execute.make_str(config, "config")
  gradients = _ops.convert_n_to_tensor(gradients, _dtypes.float32)
  learning_rates = _ops.convert_n_to_tensor(learning_rates, _dtypes.float32)
  deduplication_data = _ops.convert_to_tensor(deduplication_data, _dtypes.variant)
  _inputs_flat = list(gradients) + list(learning_rates) + [deduplication_data]
  _attrs = ("NumTables", _attr_NumTables, "NumLearningRateTags",
  _attr_NumLearningRateTags, "config", config)
  _result = _execute.execute(b"XlaSendTPUEmbeddingGradients", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result

