# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: op_py_no_precompile.cc
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

@_dispatch.add_dispatch_list
@tf_export('simple_ml_create_model_resource')
def simple_ml_create_model_resource(container="", shared_name="", name=None):
  r"""Creates a model resource and returns the handle.

  Args:
    container: An optional `string`. Defaults to `""`. Name of the container.
    shared_name: An optional `string`. Defaults to `""`.
      Name of the possibly shared name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
    Boolean feature values. Tensor of shape "batch x
    boolean_features_dim" and type float32. "Quiet Nan" represents missing
    values.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleMLCreateModelResource", name, "container", container,
        "shared_name", shared_name)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return simple_ml_create_model_resource_eager_fallback(
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      result = _dispatch.dispatch(
            simple_ml_create_model_resource, (), dict(container=container,
                                                      shared_name=shared_name,
                                                      name=name)
          )
      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return result
      raise
  # Add nodes to the TensorFlow graph.
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleMLCreateModelResource", container=container,
                                       shared_name=shared_name, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          simple_ml_create_model_resource, (), dict(container=container,
                                                    shared_name=shared_name,
                                                    name=name)
        )
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SimpleMLCreateModelResource", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SimpleMLCreateModelResource = tf_export("raw_ops.SimpleMLCreateModelResource")(_ops.to_raw_op(simple_ml_create_model_resource))


def simple_ml_create_model_resource_eager_fallback(container, shared_name, name, ctx):
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("container", container, "shared_name", shared_name)
  _result = _execute.execute(b"SimpleMLCreateModelResource", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SimpleMLCreateModelResource", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

_SimpleMLInferenceOpOutput = collections.namedtuple(
    "SimpleMLInferenceOp",
    ["dense_predictions", "dense_col_representation"])


@_dispatch.add_dispatch_list
@tf_export('simple_ml_inference_op')
def simple_ml_inference_op(numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2, model_identifier, dense_output_dim, name=None):
  r"""Applies a model and returns its predictions.

  This OP expects for a model to be loaded (e.g. by "LoadModelFromPath") before it
  is called.

  This OP expects for the input features to be flatten together, by type, as done
  by the "_InferenceArgsBuilder" utility class in "tf_op.py". For example,
  "numerical_features[i,j]" is the "j-th" numerical feature input of the model for
  the "i-th "example in the batch.

  Args:
    numerical_features: A `Tensor` of type `float32`.
      Numerical feature values. Tensor of shape "batch x
      numerical_features_dim" and type float32. "Quiet Nan" represents missing
      values.
    boolean_features: A `Tensor` of type `float32`.
      Boolean feature values. Tensor of shape "batch x
      boolean_features_dim" and type float32. "Quiet Nan" represents missing
      values.
    categorical_int_features: A `Tensor` of type `int32`.
      Categorical features stored as int. Tensor of shape
        "batch x categorical_int_features_dim" and type int32. -1 represents a missing
        value. 0 represents an "out of vocabulary" value (when applicable).

      categorical_set_int_features_{values,dim_1,dim_2}: The value and two dimension
        index set of a ragged tensor of shape "batch x num_categorical_set_features x
        num_items" i.e "x.values, x.values.row_splits and x.row_splits" respectively.
        For a given feature and example, [-1] represents a missing value.
    categorical_set_int_features_values: A `Tensor` of type `int32`.
    categorical_set_int_features_row_splits_dim_1: A `Tensor` of type `int64`.
    categorical_set_int_features_row_splits_dim_2: A `Tensor` of type `int64`.
    model_identifier: A `string`.
      Unique identifier of the model corresponding to a previously
      loaded model.
    dense_output_dim: An `int` that is `>= 1`.
      Dimension of the model output. For regression,
      dense_output_dim is the output dimension (e.g. 1 for uni-dimensional
      regression). For classification, dense_output_dim is the number of classes.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dense_predictions, dense_col_representation).

    dense_predictions: A `Tensor` of type `float32`. Tensor of shape [batch x dense_output_dim] of type float32.
      Contains a probability for classification, and a value for regression and
      ranking.
    dense_col_representation: A `Tensor` of type `string`. Tensor of shape [dense_output_dim] of type bytes.
      Contains the representation of the columns of the predictions output. For
      classification with string label, contains the name of the labels. For all
      the other cases, contains empty strings.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleMLInferenceOp", name, numerical_features,
        boolean_features, categorical_int_features,
        categorical_set_int_features_values,
        categorical_set_int_features_row_splits_dim_1,
        categorical_set_int_features_row_splits_dim_2, "model_identifier",
        model_identifier, "dense_output_dim", dense_output_dim)
      _result = _SimpleMLInferenceOpOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return simple_ml_inference_op_eager_fallback(
          numerical_features, boolean_features, categorical_int_features,
          categorical_set_int_features_values,
          categorical_set_int_features_row_splits_dim_1,
          categorical_set_int_features_row_splits_dim_2,
          model_identifier=model_identifier,
          dense_output_dim=dense_output_dim, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      result = _dispatch.dispatch(
            simple_ml_inference_op, (), dict(numerical_features=numerical_features,
                                             boolean_features=boolean_features,
                                             categorical_int_features=categorical_int_features,
                                             categorical_set_int_features_values=categorical_set_int_features_values,
                                             categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                                             categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                                             model_identifier=model_identifier,
                                             dense_output_dim=dense_output_dim,
                                             name=name)
          )
      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return result
      raise
  # Add nodes to the TensorFlow graph.
  model_identifier = _execute.make_str(model_identifier, "model_identifier")
  dense_output_dim = _execute.make_int(dense_output_dim, "dense_output_dim")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleMLInferenceOp", numerical_features=numerical_features,
                               boolean_features=boolean_features,
                               categorical_int_features=categorical_int_features,
                               categorical_set_int_features_values=categorical_set_int_features_values,
                               categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                               categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                               model_identifier=model_identifier,
                               dense_output_dim=dense_output_dim, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          simple_ml_inference_op, (), dict(numerical_features=numerical_features,
                                           boolean_features=boolean_features,
                                           categorical_int_features=categorical_int_features,
                                           categorical_set_int_features_values=categorical_set_int_features_values,
                                           categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                                           categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                                           model_identifier=model_identifier,
                                           dense_output_dim=dense_output_dim,
                                           name=name)
        )
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("model_identifier", _op.get_attr("model_identifier"),
              "dense_output_dim", _op._get_attr_int("dense_output_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SimpleMLInferenceOp", _inputs_flat, _attrs, _result)
  _result = _SimpleMLInferenceOpOutput._make(_result)
  return _result

SimpleMLInferenceOp = tf_export("raw_ops.SimpleMLInferenceOp")(_ops.to_raw_op(simple_ml_inference_op))


def simple_ml_inference_op_eager_fallback(numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2, model_identifier, dense_output_dim, name, ctx):
  model_identifier = _execute.make_str(model_identifier, "model_identifier")
  dense_output_dim = _execute.make_int(dense_output_dim, "dense_output_dim")
  numerical_features = _ops.convert_to_tensor(numerical_features, _dtypes.float32)
  boolean_features = _ops.convert_to_tensor(boolean_features, _dtypes.float32)
  categorical_int_features = _ops.convert_to_tensor(categorical_int_features, _dtypes.int32)
  categorical_set_int_features_values = _ops.convert_to_tensor(categorical_set_int_features_values, _dtypes.int32)
  categorical_set_int_features_row_splits_dim_1 = _ops.convert_to_tensor(categorical_set_int_features_row_splits_dim_1, _dtypes.int64)
  categorical_set_int_features_row_splits_dim_2 = _ops.convert_to_tensor(categorical_set_int_features_row_splits_dim_2, _dtypes.int64)
  _inputs_flat = [numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2]
  _attrs = ("model_identifier", model_identifier, "dense_output_dim",
  dense_output_dim)
  _result = _execute.execute(b"SimpleMLInferenceOp", 2, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SimpleMLInferenceOp", _inputs_flat, _attrs, _result)
  _result = _SimpleMLInferenceOpOutput._make(_result)
  return _result

_SimpleMLInferenceOpWithHandleOutput = collections.namedtuple(
    "SimpleMLInferenceOpWithHandle",
    ["dense_predictions", "dense_col_representation"])


@_dispatch.add_dispatch_list
@tf_export('simple_ml_inference_op_with_handle')
def simple_ml_inference_op_with_handle(numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2, model_handle, dense_output_dim, name=None):
  r"""TODO: add doc.

  Args:
    numerical_features: A `Tensor` of type `float32`.
    boolean_features: A `Tensor` of type `float32`.
    categorical_int_features: A `Tensor` of type `int32`.
    categorical_set_int_features_values: A `Tensor` of type `int32`.
    categorical_set_int_features_row_splits_dim_1: A `Tensor` of type `int64`.
    categorical_set_int_features_row_splits_dim_2: A `Tensor` of type `int64`.
    model_handle: A `Tensor` of type `resource`.
    dense_output_dim: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dense_predictions, dense_col_representation).

    dense_predictions: A `Tensor` of type `float32`.
    dense_col_representation: A `Tensor` of type `string`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleMLInferenceOpWithHandle", name, numerical_features,
        boolean_features, categorical_int_features,
        categorical_set_int_features_values,
        categorical_set_int_features_row_splits_dim_1,
        categorical_set_int_features_row_splits_dim_2, model_handle,
        "dense_output_dim", dense_output_dim)
      _result = _SimpleMLInferenceOpWithHandleOutput._make(_result)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return simple_ml_inference_op_with_handle_eager_fallback(
          numerical_features, boolean_features, categorical_int_features,
          categorical_set_int_features_values,
          categorical_set_int_features_row_splits_dim_1,
          categorical_set_int_features_row_splits_dim_2, model_handle,
          dense_output_dim=dense_output_dim, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      result = _dispatch.dispatch(
            simple_ml_inference_op_with_handle, (), dict(numerical_features=numerical_features,
                                                         boolean_features=boolean_features,
                                                         categorical_int_features=categorical_int_features,
                                                         categorical_set_int_features_values=categorical_set_int_features_values,
                                                         categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                                                         categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                                                         model_handle=model_handle,
                                                         dense_output_dim=dense_output_dim,
                                                         name=name)
          )
      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return result
      raise
  # Add nodes to the TensorFlow graph.
  dense_output_dim = _execute.make_int(dense_output_dim, "dense_output_dim")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleMLInferenceOpWithHandle", numerical_features=numerical_features,
                                         boolean_features=boolean_features,
                                         categorical_int_features=categorical_int_features,
                                         categorical_set_int_features_values=categorical_set_int_features_values,
                                         categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                                         categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                                         model_handle=model_handle,
                                         dense_output_dim=dense_output_dim,
                                         name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          simple_ml_inference_op_with_handle, (), dict(numerical_features=numerical_features,
                                                       boolean_features=boolean_features,
                                                       categorical_int_features=categorical_int_features,
                                                       categorical_set_int_features_values=categorical_set_int_features_values,
                                                       categorical_set_int_features_row_splits_dim_1=categorical_set_int_features_row_splits_dim_1,
                                                       categorical_set_int_features_row_splits_dim_2=categorical_set_int_features_row_splits_dim_2,
                                                       model_handle=model_handle,
                                                       dense_output_dim=dense_output_dim,
                                                       name=name)
        )
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("dense_output_dim", _op._get_attr_int("dense_output_dim"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SimpleMLInferenceOpWithHandle", _inputs_flat, _attrs, _result)
  _result = _SimpleMLInferenceOpWithHandleOutput._make(_result)
  return _result

SimpleMLInferenceOpWithHandle = tf_export("raw_ops.SimpleMLInferenceOpWithHandle")(_ops.to_raw_op(simple_ml_inference_op_with_handle))


def simple_ml_inference_op_with_handle_eager_fallback(numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2, model_handle, dense_output_dim, name, ctx):
  dense_output_dim = _execute.make_int(dense_output_dim, "dense_output_dim")
  numerical_features = _ops.convert_to_tensor(numerical_features, _dtypes.float32)
  boolean_features = _ops.convert_to_tensor(boolean_features, _dtypes.float32)
  categorical_int_features = _ops.convert_to_tensor(categorical_int_features, _dtypes.int32)
  categorical_set_int_features_values = _ops.convert_to_tensor(categorical_set_int_features_values, _dtypes.int32)
  categorical_set_int_features_row_splits_dim_1 = _ops.convert_to_tensor(categorical_set_int_features_row_splits_dim_1, _dtypes.int64)
  categorical_set_int_features_row_splits_dim_2 = _ops.convert_to_tensor(categorical_set_int_features_row_splits_dim_2, _dtypes.int64)
  model_handle = _ops.convert_to_tensor(model_handle, _dtypes.resource)
  _inputs_flat = [numerical_features, boolean_features, categorical_int_features, categorical_set_int_features_values, categorical_set_int_features_row_splits_dim_1, categorical_set_int_features_row_splits_dim_2, model_handle]
  _attrs = ("dense_output_dim", dense_output_dim)
  _result = _execute.execute(b"SimpleMLInferenceOpWithHandle", 2,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SimpleMLInferenceOpWithHandle", _inputs_flat, _attrs, _result)
  _result = _SimpleMLInferenceOpWithHandleOutput._make(_result)
  return _result


@_dispatch.add_dispatch_list
@tf_export('simple_ml_load_model_from_path')
def simple_ml_load_model_from_path(path, model_identifier, name=None):
  r"""Loads (and possibly compiles/optimizes) an Yggdrasil model in memory.

  The model is then accessible in the "kModelContainer/{model_identifier}" TF
  resource. If a model with the same "model_identifier" exists when this OP is
  called (either from the same OP instance, or from another instance with the same
  "model_identifier"), the model is discarded and replaced with the new model.

  Args:
    path: A `Tensor` of type `string`.
      Path to the Yggdrasil model. Note: a Yggdrasil model directory should
        contains a "header.pb" file.

      Returns a type-less OP that loads the model when called.
    model_identifier: A `string`.
      Unique identifier of the model. Used to create the name of
      the tf resource containing the model.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleMLLoadModelFromPath", name, path, "model_identifier",
        model_identifier)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return simple_ml_load_model_from_path_eager_fallback(
          path, model_identifier=model_identifier, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      result = _dispatch.dispatch(
            simple_ml_load_model_from_path, (), dict(path=path,
                                                     model_identifier=model_identifier,
                                                     name=name)
          )
      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return result
      raise
  # Add nodes to the TensorFlow graph.
  model_identifier = _execute.make_str(model_identifier, "model_identifier")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleMLLoadModelFromPath", path=path,
                                     model_identifier=model_identifier,
                                     name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          simple_ml_load_model_from_path, (), dict(path=path,
                                                   model_identifier=model_identifier,
                                                   name=name)
        )
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
SimpleMLLoadModelFromPath = tf_export("raw_ops.SimpleMLLoadModelFromPath")(_ops.to_raw_op(simple_ml_load_model_from_path))


def simple_ml_load_model_from_path_eager_fallback(path, model_identifier, name, ctx):
  model_identifier = _execute.make_str(model_identifier, "model_identifier")
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [path]
  _attrs = ("model_identifier", model_identifier)
  _result = _execute.execute(b"SimpleMLLoadModelFromPath", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


@_dispatch.add_dispatch_list
@tf_export('simple_ml_load_model_from_path_with_handle')
def simple_ml_load_model_from_path_with_handle(model_handle, path, name=None):
  r"""Applies a model and returns its predictions.

  Similar to "SimpleMLLoadModelFromPath", but takes a resource handle instead of
  a resource name.

  Args:
    model_handle: A `Tensor` of type `resource`.
    path: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "SimpleMLLoadModelFromPathWithHandle", name, model_handle, path)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      return simple_ml_load_model_from_path_with_handle_eager_fallback(
          model_handle, path, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      result = _dispatch.dispatch(
            simple_ml_load_model_from_path_with_handle, (), dict(model_handle=model_handle,
                                                                 path=path,
                                                                 name=name)
          )
      if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return result
      raise
  # Add nodes to the TensorFlow graph.
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SimpleMLLoadModelFromPathWithHandle", model_handle=model_handle,
                                               path=path, name=name)
  except (TypeError, ValueError):
    result = _dispatch.dispatch(
          simple_ml_load_model_from_path_with_handle, (), dict(model_handle=model_handle,
                                                               path=path,
                                                               name=name)
        )
    if result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return result
    raise
  return _op
SimpleMLLoadModelFromPathWithHandle = tf_export("raw_ops.SimpleMLLoadModelFromPathWithHandle")(_ops.to_raw_op(simple_ml_load_model_from_path_with_handle))


def simple_ml_load_model_from_path_with_handle_eager_fallback(model_handle, path, name, ctx):
  model_handle = _ops.convert_to_tensor(model_handle, _dtypes.resource)
  path = _ops.convert_to_tensor(path, _dtypes.string)
  _inputs_flat = [model_handle, path]
  _attrs = None
  _result = _execute.execute(b"SimpleMLLoadModelFromPathWithHandle", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result

