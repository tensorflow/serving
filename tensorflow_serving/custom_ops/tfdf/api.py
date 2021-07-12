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

"""Efficient inference of Yggdrasil models in tensorflow.

This API allows to run, in tensorflow, Yggdrasil models trained without the
TensorFlow Decision Forests API (i.e. generally for advanced users). More models
trained with the TensorFlow Decision Forests, you can use the SavedModel API
directory.

Note: An Yggdrasil model is a sub-part of a TensorFlow Decision Forests model.
A Yggdrasil model is always stored in the assets sub-directory
of a TensorFlow Decision Forests model. This API can effectively be used to run
TensorFlow Decision Forests models.

Usage example
=============

  # With Model(V1) in TF1

  features = {
    "a": tf.placeholder(tf.float32, [None]),
    "b": tf.placeholder(tf.string, [None])
    "c": tf.ragged.placeholder(tf.string, ragged_rank=1, value_shape=None)
    }

  model = tf_op.Model(model_path)
  predictions = model.apply(features)

  with self.session() as sess:
    # Effectively loads the model.
    sess.run(model.init_op())

    probabilities, classes = sess.run([
        predictions.dense_predictions, model_output.dense_col_representation
    ], {features["a"] : [1, 2, 3],
        features["b"] : ["a", "b", "c"],
        features["c"] : tf.ragged.constant_value(
          [["x"], ["y"], ["y", "z"], [""]], dtype=tf.string
          )})

  # With Model(V1) in TF2
  model = tf_op.Model(model_path)

  @tf.function
  def init_model():
    # Effectively loads the model.
    model.init_op()

  @tf.function
  def apply_model(features):
    return model.apply(features)

  init_model()

  features = {
    features["a"] : [1, 2, 3],
    features["b"] : ["a", "b", "c"],
    features["c"] : tf.ragged.constant(
      [["x"], ["y"], ["y", "z"], [""]], dtype=tf.string
      )
    }
  # Note: "tf.ragged.constant" in TF2 is equivalent to
  # "tf.ragged.constant_value" in TF1.

  predictions = apply_model(features)

  # With ModelV2 in TF2

  # The model is loaded in the constructor.
  model = tf_op.ModelV2(model_path)

  features = {
    features["a"] : [1, 2, 3],
    features["b"] : ["a", "b", "c"],
    features["c"] : tf.ragged.constant(
      [["x"], ["y"], ["y", "z"], [""]], dtype=tf.string
      )
    }

  # Eager predictions.
  predictions = model.apply(features)

  # Non-eager predictions.
  @tf.function
  def apply_non_eager(features):
    return model.apply(features)

  predictions_non_eager = apply_non_eager(features)

See :tf_op_test and :tf_op_tf2_test for other usage examples.

Inference OP inputs
===================

Important: Missing values should be provided as missing. A missing value is not
a value. Instead it is the absence of value. While missing value are represented
using a special value (which depend on the feature type), they are handled very
differently under the hood.

Note: A "missing value", a "out-of-vocabulary" value and an "empty set" (in the
case of a set-type feature) are three different objects.

  - All the input features of the model should be given as input tensors.
  - Numerical features are handled as float32, but can be provided as
    float{32,64} or int[32,64}. Missing numerical values should be provided as
    "quiet NaN".
  - Boolean features are represented as float32, but can also be given as
    float{32,64} or int{32,64}. Missing boolean values should be provided as
    "quiet NaN".
  - Categorical features are handled as int32 (if pre-integerized) or bytes
    (if not pre-integerized). Pre-integerized can be provided as int{32,64}.
    Missing categorical values should be provided as -1 (for integer
    categorical) or "" (empty string; for string categorical). Out of vocabulary
    should be provided as 0 (for integer categorical) or any
    string-not-int-the-dictionary (for string categorical).
  - Numerical, boolean, and categorical features are provided as dense tensor of
    shape [batch size] or [batch size, 1].
  - CategoricalSet features are handled as int32 (if pre-integerized) or bytes
    (if not pre-integerized). Pre-integerized can be provided as int{32,64}.
    Missing categorical values should be provided by [-1] (for integer
    categorical) or [""] (for string categorical). Out of vocabulary items are
    provided as 0 (for integer categorical) or any feature are provided as
    ragged tensor of shape [batch size, num items].

Inference OP outputs
====================

  Classification:
    dense_predictions: float32 tensor of probabilities of shape [batch size,
      num classes].
    dense_col_representation: string tensor of shape [num classes].
      Representation of the classes.

  Regression:
    dense_predictions: float32 tensor of regressive values of shape [batch size,
      1].
    dense_col_representation: string tensor of shape [1]. Contains
      only empty values.

Categorical features
====================

Unlike the tf.estimator and Keras API, Yggdrasil differentiates between
categorical, categorical-set and categorical-list features. Make sure to use the
correct one for your case. All three types support "missing values" (which is
semantically different from being empty, in the case of categorical-set and
categorical-list).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os
from typing import Text, Dict, List, Any, Optional
import uuid
from absl import logging
import six

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.training.tracking import base as trackable_base
from tensorflow.python.training.tracking import tracking
# pylint: enable=g-direct-tensorflow-import

from tensorflow_decision_forests.tensorflow.ops.inference import op
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
tf.load_op_library(resource_loader.get_path_to_datafile("inference.so"))

Tensor = Any
InitOp = Tensor
Task = abstract_model_pb2.Task
ColumnType = data_spec_pb2.ColumnType

# Wrapper around the outputs values of the inference op.
ModelOutput = collections.namedtuple(
    "ModelOutput",
    [
        # Predictions of the model e.g. probabilities. See the documentation of
        # "dense_predictions" in "inference_interface.cc" for the format
        # details (type, shape, semantic).
        "dense_predictions",

        # String representation of the model output. See the documentation
        # of "dense_col_representation" in "inference_interface.cc" for the
        # format details (type, shape, semantic).
        "dense_col_representation",
    ])

# Magic value used to indicate of a missing value for categorical stored as
# ints, but that should not be interpreted as integer directly.
#
# Note: TF estimators don't standardize missing value representation.
MISSING_NON_INTEGERIZED_CATEGORICAL_STORED_AS_INT = 0x7FFFFFFF - 2


class Model(object):
  """Applies a Yggdrasil model.

  For TensorFlow V1 and tensorflow V2.

  If you are using tensorflow v2 and if you want the model serialization to
  Assets to be handled automatically, use "ModelV2" instead.
  """

  def __init__(self,
               model_path: Text,
               tensor_model_path: Optional[Tensor] = None,
               verbose: Optional[bool] = True):
    """Initialize the model.

    The Yggdrasil model should be available at the "model_path" location both at
    the "Model" object creation, and during the call to "init_op()".

    Args:
      model_path: Path to the Yggdrasil model.
      tensor_model_path: Path of the model at execution time. If no provided,
        will use "model_path" instead. This argument can be use to load model
        from SavedModel assets.
      verbose: If true, prints information about the model and its integration
        in tensorflow.
    """

    self._verbose: Optional[bool] = verbose

    if self._verbose:
      logging.info("Create inference model for %s", model_path)

    # Identifier of the model tf resource. Allow a same model to be applied on
    # separate inputs tensors.
    self.model_identifier = _create_model_identifier()

    self.input_builder = _InferenceArgsBuilder(verbose)
    self.input_builder.build_from_model_path(model_path)

    # Model loading and initialization op.
    if tensor_model_path is None:
      tensor_model_path = model_path
    load_model_op = op.SimpleMLLoadModelFromPath(
        model_identifier=self.model_identifier, path=tensor_model_path)

    self._init_op = tf.group(self.input_builder.init_op(), load_model_op)

  def init_op(self) -> InitOp:
    """Get the model "init_op".

    This op initializes the model (effectively loading it in memory). This op
    should be called before the model is applied.

    Returns:
      The init_op.
    """
    return self._init_op

  def apply(self, features: Dict[Text, Tensor]) -> ModelOutput:
    """Applies the model.

    Args:
      features: Dictionary of input features of the model. All the input
        features of the model should be available. Features not used by the
        model are ignored.

    Returns:
      Predictions of the model.
    """

    if self._verbose:
      logging.info("Create inference op")

    inference_args = self.input_builder.build_inference_op_args(features)
    dense_predictions, dense_col_representation = op.SimpleMLInferenceOp(
        model_identifier=self.model_identifier, **inference_args)

    return ModelOutput(
        dense_predictions=dense_predictions,
        dense_col_representation=dense_col_representation)


class ModelV2(tracking.AutoTrackable):
  """Applies a Yggdrasil model.

  For TensorFlow V2.
  """

  def __init__(self, model_path: Text, verbose: Optional[bool] = True):
    """Initialize the model.

    The model content will be serialized as an asset if necessary.

    Args:
      model_path: Path to the Yggdrasil model.
      verbose: Should details about the calls be printed.
    """

    super(ModelV2).__init__()
    self._input_builder = _InferenceArgsBuilder(verbose)
    self._input_builder.build_from_model_path(model_path)
    self._compiled_model = _CompiledSimpleMLModelResource(
        _DiskModelLoader(model_path))

  def apply(self, features: Dict[Text, Tensor]) -> ModelOutput:
    """Applies the model.

    Args:
      features: Dictionary of input features of the model. All the input
        features of the model should be available. Features not used by the
        model are ignored.

    Returns:
      Predictions of the model.
    """

    inference_args = self._input_builder.build_inference_op_args(features)

    (dense_predictions,
     dense_col_representation) = op.SimpleMLInferenceOpWithHandle(
         model_handle=self._compiled_model.resource_handle,
         name="inference_op",
         **inference_args)

    return ModelOutput(
        dense_predictions=dense_predictions,
        dense_col_representation=dense_col_representation)


def _create_model_identifier() -> Text:
  """Creates a unique identifier for the model.

  This identifier is used internally by the library.

  Returns:
    String identifier.
  """
  return "sml_{}".format(uuid.uuid4())


# For each type of features, a map between a feature index (from
# "_feature_name_to_idx") and the input tensor for this feature.
FeatureMaps = collections.namedtuple("FeatureMaps", [
    "numerical_features",
    "boolean_features",
    "categorical_int_features",
    "categorical_set_int_features",
])


class _InferenceArgsBuilder(tracking.AutoTrackable):
  """Utility for the creation of the argument of the inference OP."""

  def __init__(self, verbose: Optional[bool] = True):

    super(_InferenceArgsBuilder).__init__()
    self._verbose: bool = verbose
    self._header: Optional[abstract_model_pb2.AbstractModel] = None
    self._data_spec: Optional[data_spec_pb2.DataSpecification] = None
    self._feature_name_to_idx = None

    # List of initialization ops.
    self._init_ops: List[tf.Operation] = None

    # How many dimensions has the model predictions.
    self._dense_output_dim: Optional[int] = None

    super(_InferenceArgsBuilder, self).__init__()

  def build_from_model_path(self, model_path: Text):
    # Load model meta-data.
    header = abstract_model_pb2.AbstractModel()
    with tf.io.gfile.GFile(os.path.join(model_path, "header.pb"), "rb") as f:
      header.ParseFromString(f.read())

    data_spec = data_spec_pb2.DataSpecification()
    with tf.io.gfile.GFile(os.path.join(model_path, "data_spec.pb"), "rb") as f:
      data_spec.ParseFromString(f.read())

    self.build_from_dataspec_and_header(data_spec, header)

  def build_from_dataspec_and_header(self,
                                     dataspec: data_spec_pb2.DataSpecification,
                                     header: abstract_model_pb2.AbstractModel):
    self._header = header
    self._data_spec = dataspec

    # Map between the input feature names and their indices.
    self._feature_name_to_idx = {
        self._data_spec.columns[feature_idx].name: feature_idx
        for feature_idx in self._header.input_features
    }

    self._init_ops = []
    self._dense_output_dim = self._get_dense_output_dim()

    self._create_str_to_int_tables()

  def init_op(self) -> Tensor:
    """Op initializing the processing of the input features."""

    if self._init_ops:
      return tf.group(*self._init_ops)
    else:
      return tf.no_op()

  def build_inference_op_args(self, features: Dict[Text,
                                                   Tensor]) -> Dict[Text, Any]:
    """Creates the arguments of the SimpleMLInferenceOp.

    Args:
      features: Dictionary of input features of the model. All the input
        features of the model should be available. Features not used by the
        model are ignored.

    Returns:
      Op constructor arguments.
    """

    if self._verbose:
      logging.info("\tApply model on features:\n%s", features)

    # Extract, clean, check and index the input feature tensors.
    feature_maps = FeatureMaps(
        numerical_features={},
        boolean_features={},
        categorical_int_features={},
        categorical_set_int_features={})

    for feature_name, feature_tensor in features.items():
      self._register_input_feature(feature_name, feature_tensor, feature_maps)

    self._check_all_input_features_are_provided(feature_maps)

    # Pack the input features by type.

    # Numerical features.
    if feature_maps.numerical_features:
      numerical_features = tf.stack(
          self._dict_to_list_sorted_by_key(feature_maps.numerical_features),
          axis=1)
    else:
      numerical_features = tf.constant(0, dtype=tf.float32, shape=(0, 0))

    # Boolean features.
    if feature_maps.boolean_features:
      boolean_features = tf.stack(
          self._dict_to_list_sorted_by_key(feature_maps.boolean_features),
          axis=1)
    else:
      boolean_features = tf.constant(0, dtype=tf.float32, shape=(0, 0))

    # Categorical features.
    if feature_maps.categorical_int_features:
      categorical_int_features = tf.stack(
          self._dict_to_list_sorted_by_key(
              feature_maps.categorical_int_features),
          axis=1)
    else:
      categorical_int_features = tf.constant(0, dtype=tf.int32, shape=(0, 0))

    # Categorical Set features.
    if feature_maps.categorical_set_int_features:
      categorical_set_int_features = tf.stack(
          self._dict_to_list_sorted_by_key(
              feature_maps.categorical_set_int_features),
          axis=1)

    else:
      categorical_set_int_features = tf.ragged.constant([],
                                                        dtype=tf.int32,
                                                        ragged_rank=2)

    args = {
        "numerical_features":
            numerical_features,
        "boolean_features":
            boolean_features,
        "categorical_int_features":
            categorical_int_features,
        "categorical_set_int_features_values":
            categorical_set_int_features.values.values,
        "categorical_set_int_features_row_splits_dim_1":
            categorical_set_int_features.values.row_splits,
        "categorical_set_int_features_row_splits_dim_2":
            categorical_set_int_features.row_splits,
        "dense_output_dim":
            self._dense_output_dim,
    }

    if self._verbose:
      logging.info("Inference op arguments:\n%s", args)

    return args

  def _register_input_feature(self, name: Text, value: Tensor,
                              feature_maps: FeatureMaps) -> None:
    """Indexes, and optionally pre-computes, the input feature tensors.

    Args:
      name: Name of the input feature.
      value: Tensor value of the input feature.
      feature_maps: Output index of input features.

    Raises:
      Exception: Is the feature is already registered, or with the wrong format.
    """

    feature_idx = self._feature_name_to_idx.get(name)
    if feature_idx is None:
      logging.warn("Registering feature \"%s\" not used by the model.", name)
      return

    if feature_idx in self._all_feature_idxs(feature_maps):
      raise Exception("The feature \"{}\" was already registered.".format(name))

    feature_spec = self._data_spec.columns[feature_idx]
    if feature_spec.type == ColumnType.NUMERICAL:
      value = self._prepare_and_check_numerical_feature(name, value)
      feature_maps.numerical_features[feature_idx] = value

    elif feature_spec.type == ColumnType.BOOLEAN:
      value = self._prepare_and_check_boolean_feature(name, value)
      feature_maps.boolean_features[feature_idx] = value

    elif feature_spec.type == ColumnType.CATEGORICAL:
      value = self._prepare_and_check_categorical_feature(
          name, value, feature_spec)
      feature_maps.categorical_int_features[feature_idx] = value

    elif feature_spec.type == ColumnType.CATEGORICAL_SET:
      value = self._prepare_and_check_categorical_set_feature(
          name, value, feature_spec)
      feature_maps.categorical_set_int_features[feature_idx] = value

    else:
      raise Exception("No supported type \"{}\" for feature \"{}\"".format(
          ColumnType.Name(feature_spec.type), name))

  def _create_str_to_int_tables(self):
    """Creates the tables used to convert categorical features into integers."""

    # Map from feature index to the string->int hashmap.
    self.categorical_str_to_int_hashmaps = {}
    for feature_idx in self._header.input_features:
      feature_spec = self._data_spec.columns[feature_idx]
      if feature_spec.HasField(
          "categorical"
      ) and not feature_spec.categorical.is_already_integerized:
        # Extract the vocabulary of the feature.
        #
        # Note: The item with index "0" is the "out of vocabulary". It is
        # handled by the hashmap directly.
        vocabulary = [(key, item.index)
                      for key, item in feature_spec.categorical.items.items()
                      if item.index != 0]
        # Missing value.
        vocabulary.append(("", -1))
        vocabulary.append(
            (str(MISSING_NON_INTEGERIZED_CATEGORICAL_STORED_AS_INT), -1))
        vocabulary.sort(key=lambda x: x[1])

        # Create a hasmap table with the vocabulary.
        vocabulary_keys = tf.constant(list(zip(*vocabulary))[0])
        vocabulary_values = tf.constant(list(zip(*vocabulary))[1])
        vocabulary_index = tf.lookup.KeyValueTensorInitializer(
            vocabulary_keys, vocabulary_values)
        # Note: Value "0" is the out-of-vocabulary.
        vocabulary_hashmap = tf.lookup.StaticHashTable(vocabulary_index, 0)

        self._init_ops.append(vocabulary_index.initialize(vocabulary_hashmap))

        self.categorical_str_to_int_hashmaps[
            feature_spec.name] = vocabulary_hashmap

  @staticmethod
  def _dict_to_list_sorted_by_key(src: Dict[Any, Any]) -> List[Any]:
    """Extracts the values of a dictionary, sorted by key values.

    Examples:
      {2:"b", 3:"c", 1:"a"} -> ["a", "b", "c"]

    Args:
      src: Dictionary to process.

    Returns:
      Input values sorted by key.
    """

    return [value[1] for value in sorted(src.items())]

  @staticmethod
  def _all_feature_idxs(feature_maps: FeatureMaps):
    """Lists all the input feature indices."""
    idxs = []
    for field_name in feature_maps._fields:
      idxs.extend(getattr(feature_maps, field_name).keys())
    return idxs

  def _check_all_input_features_are_provided(self, feature_maps):
    """Making sure all the input features of the model are provided."""

    missing_features = set(self._feature_name_to_idx.values()).difference(
        set(self._all_feature_idxs(feature_maps)))
    if missing_features:
      raise Exception(
          "No all input features have been registered. Non registered required "
          "input features: {}".format([
              self._data_spec.columns[feature_idx].name
              for feature_idx in missing_features
          ]))

  def _get_dense_output_dim(self):
    """Gets the dimension of the op output."""

    label_spec = self._data_spec.columns[self._header.label_col_idx]
    if self._header.task == Task.CLASSIFICATION:
      return label_spec.categorical.number_of_unique_values - 1
    elif self._header.task == Task.REGRESSION:
      return 1
    elif self._header.task == Task.RANKING:
      return 1
    else:
      raise Exception("Non supported task {}.".format(
          Task.Name(self._header.task)))

  def _prepare_and_check_numerical_feature(self, name: Text, value: Tensor):
    """Checks and optionally pre-processes a numerical feature."""

    extended_name = "Numerical feature \"{}\"".format(name)
    if value.dtype not in [tf.float32, tf.int32, tf.int64, tf.float64]:
      raise Exception(
          "{} is expected to have type float{{32,64}} or int{{32,64}}. Got {} "
          "instead".format(extended_name, value.dtype))

    if value.dtype != tf.float32:
      value = tf.cast(value, tf.float32)

    if len(value.shape) == 2:
      if value.shape[1] != 1:
        raise Exception(
            "{} is expected to have shape [None] or [None, 1]. Got {}  instead."
            .format(extended_name, len(value.shape)))
      value = value[:, 0]

    elif len(value.shape) != 1:
      raise Exception(
          "{} is expected to have shape [None] or [None, 1]. Got {}  instead."
          .format(extended_name, len(value.shape)))
    return value

  def _prepare_and_check_boolean_feature(self, name: Text, value: Tensor):
    """Checks and optionally pre-processes a boolean feature."""

    extended_name = "Boolean feature \"{}\"".format(name)
    if value.dtype not in [tf.float32, tf.int32, tf.int64, tf.float64]:
      raise Exception(
          "{} is expected to have type float{{32,64}} or int{{32,64}}. Got {} "
          "instead".format(extended_name, value.dtype))

    if value.dtype != tf.float32:
      value = tf.cast(value, tf.float32)

    if len(value.shape) == 2:
      if value.shape[1] != 1:
        raise Exception(
            "{} is expected to have shape [None] or [None, 1]. Got {}  instead."
            .format(extended_name, len(value.shape)))
      value = value[:, 0]

    elif len(value.shape) != 1:
      raise Exception(
          "{} is expected to have shape [None] or [None, 1]. Got {}  instead."
          .format(extended_name, len(value.shape)))
    return value

  def _prepare_and_check_categorical_feature(
      self, name: Text, value: Tensor,
      feature_spec: data_spec_pb2.Column) -> Tensor:
    """Checks and optionally pre-processes a categorical feature.

    Args:
      name: Name of the feature.
      value: Tensor value of the feature.
      feature_spec: Feature spec (e.g. type, dictionary, statistics) of the
        feature.

    Returns:
      The feature value ready to be consumed by the inference op.

    Raises:
      Exception: In case of unexpected feature type or shape.
    """

    extended_name = "Categorical feature \"{}\"".format(name)

    if value.dtype in [tf.int32, tf.int64]:
      # Native format.
      if not feature_spec.categorical.is_already_integerized:
        # A categorical feature, stored as integer, but not already integerized.
        value = self.categorical_str_to_int_hashmaps[name].lookup(
            tf.strings.as_string(value))

      if value.dtype != tf.int32:
        value = tf.cast(value, tf.int32)

    elif value.dtype == tf.string:
      if feature_spec.categorical.is_already_integerized:
        raise Exception(
            "{} was feed as {}. Expecting int32 tensor instead.".format(
                extended_name, value))

      value = self.categorical_str_to_int_hashmaps[name].lookup(value)

    else:
      raise Exception(
          "{} is expected to have type int32, int64 or string. Got {} instead"
          .format(extended_name, value.dtype))

    if len(value.shape) == 2:
      if value.shape[1] != 1:
        raise Exception(
            "{} is expected to have shape [None] or [None, 1]. Got {}  instead."
            .format(extended_name, len(value.shape)))
      value = value[:, 0]
    elif len(value.shape) != 1:
      raise Exception("{} is expected to have rank 1. Got {}  instead.".format(
          extended_name, len(value.shape)))

    return value

  def _prepare_and_check_categorical_set_feature(
      self, name: Text, value: Tensor,
      feature_spec: data_spec_pb2.Column) -> Tensor:
    """Checks and optionally pre-processes a categorical set feature.

    Args:
      name: Name of the feature.
      value: Tensor value of the feature.
      feature_spec: Feature spec (e.g. type, dictionary, statistics) of the
        feature.

    Returns:
      The feature value ready to be consumed by the inference op.

    Raises:
      Exception: In case of unexpected feature type or shape.
    """

    extended_name = "Categorical set feature \"{}\"".format(name)

    if not isinstance(value, tf.RaggedTensor):
      raise Exception(
          "{} was feed as {}. Expecting a RaggedTensor instead.".format(
              extended_name, value))

    if value.dtype in [tf.int32, tf.int64]:
      # Native format.
      if not feature_spec.categorical.is_already_integerized:
        raise Exception(
            "{} was feed as {}. Expecting string tensor instead.".format(
                extended_name, value))

      if value.dtype != tf.int32:
        value = tf.cast(value, tf.int32)

    elif value.dtype == tf.string:
      if feature_spec.categorical.is_already_integerized:
        raise Exception(
            "{} was feed as {}. Expecting int32 tensor instead.".format(
                extended_name, value))

      value = tf.ragged.map_flat_values(
          self.categorical_str_to_int_hashmaps[name].lookup, value)

    else:
      raise Exception(
          "{} is expected to have type int32, int64 or string. Got {} instead"
          .format(extended_name, value.dtype))

    return value


class _AbstractModelLoader(six.with_metaclass(abc.ABCMeta, object)):
  """Loads a model in a _CompiledSimpleMLModelResource."""

  @abc.abstractmethod
  def initialize(self, model: "_CompiledSimpleMLModelResource") -> tf.Operation:
    raise NotImplementedError()


class _CompiledSimpleMLModelResource(tracking.TrackableResource):
  """Utility class to handle compiled model resources.

  This code is directly copied from StaticHashTable in:
    google3/third_party/tensorflow/python/ops/lookup_ops.py
  """

  def __init__(self, model_loader: _AbstractModelLoader):

    super(_CompiledSimpleMLModelResource, self).__init__()

    if isinstance(model_loader, trackable_base.Trackable):
      self._model_loader = self._track_trackable(model_loader, "_model_loader")

    self._shared_name = "simple_ml_model_%s" % (str(uuid.uuid4()),)

    with tf.init_scope():
      self._resource_handle = self._create_resource()

    if (not context.executing_eagerly() and
        tf.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with tf.init_scope():
        self._init_op = self._initialize()
    else:
      self._init_op = self._initialize()

  def _create_resource(self):
    table_ref = op.SimpleMLCreateModelResource(shared_name=self._shared_name)
    return table_ref

  def _initialize(self):
    return self._model_loader.initialize(self)


class _DiskModelLoader(_AbstractModelLoader, tracking.AutoTrackable):
  """Loads a model from disk.

  This code is directly copied from TextFileInitializer in:
    google3/third_party/tensorflow/python/ops/lookup_ops.py
  """

  def __init__(self, model_path):

    super(_DiskModelLoader).__init__()
    if not isinstance(model_path, tf.Tensor) and not model_path:
      raise ValueError("Filename required")

    self._all_files = []
    self._done_file = None
    for directory, _, filenames in tf.io.gfile.walk(model_path):
      for filename in filenames:
        path = os.path.join(directory, filename)
        asset = tf.saved_model.Asset(path)
        if filename == "done":
          self._done_file = asset
        self._all_files.append(asset)
    if self._done_file is None:
      raise ValueError(f"The model at {model_path} is invalid as it is "
                       "missing a \"done\" file.")

    super(_DiskModelLoader, self).__init__()

  def initialize(self, model: _CompiledSimpleMLModelResource) -> tf.Operation:

    model_path = tf.strings.regex_replace(self._done_file.asset_path, "done",
                                          "")
    with ops.name_scope("simple_ml", "load_model_from_disk",
                        (model.resource_handle,)):
      init_op = op.SimpleMLLoadModelFromPathWithHandle(
          model_handle=model.resource_handle, path=model_path)

    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op

  def get_model_path(self) -> Tensor:
    """Gets the path to the model on disk."""

    return tf.strings.regex_replace(self._done_file.asset_path, "done", "")
