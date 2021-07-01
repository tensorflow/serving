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

"""Utility functions to unit-test model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import tensorflow.compat.v1 as tf

from tensorflow_decision_forests.component.inspector import blob_sequence
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from yggdrasil_decision_forests.model.gradient_boosted_trees import gradient_boosted_trees_pb2
from yggdrasil_decision_forests.model.random_forest import random_forest_pb2
from yggdrasil_decision_forests.utils import distribution_pb2


def build_toy_data_spec(add_boolean_features=False, has_catset=False):
  """Creates a toy dataspec with 4 features."""

  columns = [
      data_spec_pb2.Column(name="a", type=data_spec_pb2.NUMERICAL),
      data_spec_pb2.Column(
          name="b",
          type=data_spec_pb2.CATEGORICAL,
          categorical=data_spec_pb2.CategoricalSpec(
              number_of_unique_values=3,
              items={
                  "oov": data_spec_pb2.CategoricalSpec.VocabValue(index=0),
                  "x": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                  "y": data_spec_pb2.CategoricalSpec.VocabValue(index=2),
                  "z": data_spec_pb2.CategoricalSpec.VocabValue(index=3)
              })),
      data_spec_pb2.Column(
          name="c",
          type=data_spec_pb2.CATEGORICAL,
          categorical=data_spec_pb2.CategoricalSpec(
              number_of_unique_values=5, is_already_integerized=True)),
      data_spec_pb2.Column(
          name="label",
          type=data_spec_pb2.CATEGORICAL,
          categorical=data_spec_pb2.CategoricalSpec(
              number_of_unique_values=4,
              items={
                  "oov": data_spec_pb2.CategoricalSpec.VocabValue(index=0),
                  "v1": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                  "v2": data_spec_pb2.CategoricalSpec.VocabValue(index=2),
                  "v3": data_spec_pb2.CategoricalSpec.VocabValue(index=3)
              })),
      data_spec_pb2.Column(
          name="label_binary",
          type=data_spec_pb2.CATEGORICAL,
          categorical=data_spec_pb2.CategoricalSpec(
              number_of_unique_values=3,
              items={
                  "oov": data_spec_pb2.CategoricalSpec.VocabValue(index=0),
                  "v1": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                  "v2": data_spec_pb2.CategoricalSpec.VocabValue(index=2)
              })),
  ]

  assert not (add_boolean_features and has_catset)

  if add_boolean_features:
    columns.append(
        data_spec_pb2.Column(name="bool_feature", type=data_spec_pb2.BOOLEAN))

  if has_catset:
    columns.append(
        data_spec_pb2.Column(
            name="d",
            type=data_spec_pb2.CATEGORICAL_SET,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=4,
                items={
                    "oov": data_spec_pb2.CategoricalSpec.VocabValue(index=0),
                    "x": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                    "y": data_spec_pb2.CategoricalSpec.VocabValue(index=2),
                    "z": data_spec_pb2.CategoricalSpec.VocabValue(index=3)
                })))
    columns.append(
        data_spec_pb2.Column(
            name="e",
            type=data_spec_pb2.CATEGORICAL_SET,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=20, is_already_integerized=True)))

  return data_spec_pb2.DataSpecification(columns=columns)


def build_toy_random_forest(path,
                            winner_take_all_inference,
                            add_boolean_features=False,
                            has_catset=False):
  """Creates a toy Random Forest model compatible with _build_toy_data_spec."""

  logging.info("Create toy model in %s", path)

  tf.io.gfile.makedirs(path)

  with tf.io.gfile.GFile(os.path.join(path, "done"), "w") as f:
    f.write("Something")

  data_spec = build_toy_data_spec(
      add_boolean_features=add_boolean_features, has_catset=has_catset)
  with tf.io.gfile.GFile(os.path.join(path, "data_spec.pb"), "w") as f:
    f.write(data_spec.SerializeToString())

  header = abstract_model_pb2.AbstractModel(
      name="RANDOM_FOREST",
      task=abstract_model_pb2.CLASSIFICATION,
      label_col_idx=3,
      input_features=[0, 1, 2] + ([5] if add_boolean_features else []) +
      ([5, 6] if has_catset else []))
  with tf.io.gfile.GFile(os.path.join(path, "header.pb"), "w") as f:
    f.write(header.SerializeToString())

  rf_header = random_forest_pb2.Header(
      num_node_shards=1,
      num_trees=2,
      winner_take_all_inference=winner_take_all_inference,
      node_format="BLOB_SEQUENCE")
  with tf.io.gfile.GFile(os.path.join(path, "random_forest_header.pb"),
                         "w") as f:
    f.write(rf_header.SerializeToString())

  with blob_sequence.Writer(os.path.join(
      path, "nodes-00000-of-00001")) as output_file:

    for _ in range(rf_header.num_trees):
      # [a > 1 ] // Node 0
      #   |-- [b in ["x,"y"] ] // Node 1
      #   |     |-- [label = 80%;10%;10%] // Node 2
      #   |     L-- [label = 10%;80%;10%] // Node 3
      #   L-- [c in [1, 3] ] // Node 4
      #         |-- [label = 50%;50%;0%] // Node 5
      #         L-- [label = 0%;50%;50%] // Node 6
      #
      # If add_boolean_features is True, Node 6 is repurposed as follows:
      #
      # ['bool' is True] // Node 6
      #   | -- [label = 0%;20%;80%] // Node 7
      #   L -- [label = 0%;80%;20%] // Node 8
      #
      # If has_catset is True, Node 4 condition is replaced by:
      #   [ d \intersect [1,3] != \emptyset
      # Node 0
      node = decision_tree_pb2.Node(
          condition=decision_tree_pb2.NodeCondition(
              na_value=False,
              attribute=0,
              condition=decision_tree_pb2.Condition(
                  higher_condition=decision_tree_pb2.Condition.Higher(
                      threshold=1.0)),
          ))
      output_file.write(node.SerializeToString())

      # Node 1
      node = decision_tree_pb2.Node(
          condition=decision_tree_pb2.NodeCondition(
              na_value=False,
              attribute=1,
              condition=decision_tree_pb2.Condition(
                  contains_bitmap_condition=decision_tree_pb2.Condition
                  .ContainsBitmap(elements_bitmap=b"\x06")),  # [1,2]
          ))
      output_file.write(node.SerializeToString())

      # Node 2
      node = decision_tree_pb2.Node(
          classifier=decision_tree_pb2.NodeClassifierOutput(
              top_value=1,
              distribution=distribution_pb2.IntegerDistributionDouble(
                  counts=[0, 0.8, 0.1, 0.1], sum=1)))
      output_file.write(node.SerializeToString())

      # Node 3
      node = decision_tree_pb2.Node(
          classifier=decision_tree_pb2.NodeClassifierOutput(
              top_value=2,
              distribution=distribution_pb2.IntegerDistributionDouble(
                  counts=[0, 0.1, 0.8, 0.1], sum=1)))
      output_file.write(node.SerializeToString())

      # Node 4
      node = decision_tree_pb2.Node(
          condition=decision_tree_pb2.NodeCondition(
              na_value=False,
              attribute=5 if has_catset else 2,
              condition=decision_tree_pb2.Condition(
                  contains_condition=decision_tree_pb2.Condition.ContainsVector(
                      elements=[1, 3]))))
      output_file.write(node.SerializeToString())

      # Node 5
      node = decision_tree_pb2.Node(
          classifier=decision_tree_pb2.NodeClassifierOutput(
              top_value=1,
              distribution=distribution_pb2.IntegerDistributionDouble(
                  counts=[0, 1, 1, 0], sum=2)))
      output_file.write(node.SerializeToString())

      if not add_boolean_features:
        # Node 6
        node = decision_tree_pb2.Node(
            classifier=decision_tree_pb2.NodeClassifierOutput(
                top_value=2,
                distribution=distribution_pb2.IntegerDistributionDouble(
                    counts=[0, 0, 1, 1], sum=2)))
        output_file.write(node.SerializeToString())
      else:
        # Node 6
        node = decision_tree_pb2.Node(
            condition=decision_tree_pb2.NodeCondition(
                na_value=False,
                attribute=5,
                condition=decision_tree_pb2.Condition(
                    true_value_condition=decision_tree_pb2.Condition.TrueValue(
                    ))))
        output_file.write(node.SerializeToString())

        # Node 7
        node = decision_tree_pb2.Node(
            classifier=decision_tree_pb2.NodeClassifierOutput(
                top_value=3,
                distribution=distribution_pb2.IntegerDistributionDouble(
                    counts=[0, 0, 0.2, 0.8], sum=1)))
        output_file.write(node.SerializeToString())

        # Node 8
        node = decision_tree_pb2.Node(
            classifier=decision_tree_pb2.NodeClassifierOutput(
                top_value=2,
                distribution=distribution_pb2.IntegerDistributionDouble(
                    counts=[0, 0, 0.8, 0.2], sum=1)))
        output_file.write(node.SerializeToString())


def build_toy_gbdt(path, num_classes):
  """Creates a toy GBDT model compatible with _build_toy_data_spec."""

  logging.info("Create toy model in %s", path)

  tf.io.gfile.makedirs(path)

  with tf.io.gfile.GFile(os.path.join(path, "done"), "w") as f:
    f.write("Something")

  data_spec = build_toy_data_spec()
  with tf.io.gfile.GFile(os.path.join(path, "data_spec.pb"), "w") as f:
    f.write(data_spec.SerializeToString())

  header = abstract_model_pb2.AbstractModel(
      name="GRADIENT_BOOSTED_TREES",
      task=abstract_model_pb2.CLASSIFICATION,
      label_col_idx=4 if num_classes == 2 else 3,
      input_features=[0, 1, 2])
  with tf.io.gfile.GFile(os.path.join(path, "header.pb"), "w") as f:
    f.write(header.SerializeToString())

  num_iters = 2
  num_trees_per_iter = 1 if num_classes == 2 else num_classes

  rf_header = gradient_boosted_trees_pb2.Header(
      num_node_shards=1,
      num_trees=num_iters * num_trees_per_iter,
      loss=gradient_boosted_trees_pb2.BINOMIAL_LOG_LIKELIHOOD if num_classes
      == 2 else gradient_boosted_trees_pb2.MULTINOMIAL_LOG_LIKELIHOOD,
      initial_predictions=[1.0] if num_classes == 2 else [0.0] * num_classes,
      num_trees_per_iter=num_trees_per_iter,
      node_format="BLOB_SEQUENCE")
  with tf.io.gfile.GFile(
      os.path.join(path, "gradient_boosted_trees_header.pb"), "w") as f:
    f.write(rf_header.SerializeToString())

  with blob_sequence.Writer(os.path.join(
      path, "nodes-00000-of-00001")) as output_file:

    for _ in range(num_iters):
      for tree_in_iter_idx in range(num_trees_per_iter):

        # [a > 1 ] // Node 0
        #   |-- [label = 1.0 + tree_in_iter_idx] // Node 1
        #   L-- [label = 5.0 + tree_in_iter_idx^2] // Node 2
        #
        # Two classes
        #   Case a<=1:
        #     logit = 1.0 + 1.0 * 2 = 3.0
        #     proba = [0.0474259, 0.9525741]
        #   Case a>1:
        #     logit = 1.0 + 5.0 * 2 = 11.0
        #     proba = [1.67e-05, 0.9999833]
        #
        # Three classes
        #   Case a<=1:
        #     logit = [1.0 * 2, 2.0 * 2, 3.0 * 2] = [2.0, 4.0, 6.0]
        #     proba = [0.01587624 0.11731043 0.86681333]
        #   Case a>1:
        #     logit = [5.0 * 2, 6.0 * 2, 9.0 * 2] = [10.0, 12.0, 18.0]
        #     proba = [0.01587624 0.11731043 0.86681333]

        # Node 0
        node = decision_tree_pb2.Node(
            condition=decision_tree_pb2.NodeCondition(
                na_value=False,
                attribute=0,
                condition=decision_tree_pb2.Condition(
                    higher_condition=decision_tree_pb2.Condition.Higher(
                        threshold=1.0)),
            ))
        output_file.write(node.SerializeToString())

        # Node 1
        node = decision_tree_pb2.Node(
            regressor=decision_tree_pb2.NodeRegressorOutput(top_value=1.0 +
                                                            tree_in_iter_idx))
        output_file.write(node.SerializeToString())

        # Node 2
        node = decision_tree_pb2.Node(
            regressor=decision_tree_pb2.NodeRegressorOutput(
                top_value=5.0 + tree_in_iter_idx * tree_in_iter_idx))
        output_file.write(node.SerializeToString())


def build_toy_input_features(use_rank_two=False, has_catset=False):
  """Creates tf placeholders for the toy dataset _build_toy_data_spec."""

  feature_shape = [None, 1] if use_rank_two else [None]
  features = {
      "a": tf.placeholder(tf.float32, feature_shape),
      "b": tf.placeholder(tf.string, feature_shape),
      "c": tf.placeholder(tf.int64, feature_shape),
      "bool_feature": tf.placeholder(tf.float32, feature_shape),
      "unused": tf.placeholder(tf.float32, feature_shape)
  }

  if has_catset:
    features["d"] = tf.ragged.placeholder(
        tf.string, ragged_rank=1, value_shape=None)
    features["e"] = tf.ragged.placeholder(
        tf.int32, ragged_rank=1, value_shape=None)

  return features


def build_toy_input_feature_values(features,
                                   use_rank_two=False,
                                   has_catset=False):
  """Create a set of input features values.

  These examples will fall respectively in the nodes 6, 5, 3, 2 of
  _build_toy_random_forest.

  Args:
    features: Dictionary of input feature tensors. If None, the features are
      indexed by name (used in tf2).
    use_rank_two: Should the feature be passed as one or two ranked tensors.
    has_catset: Add two categorical-set features to the dataspec.

  Returns:
    Dictionary of feature values.
  """

  is_tf2 = features is None

  def shape(x):
    if use_rank_two:
      y = [[v] for v in x]
    else:
      y = x
    if is_tf2:
      return tf.constant(y)
    else:
      return y

  if is_tf2:

    class Identity:

      def __getitem__(self, key):
        return key

    features = Identity()

  feature_values = {
      features["a"]: shape([2, 2, 0, 0]),
      features["b"]: shape(["x", "z", "x", "z"]),
      features["c"]: shape([1, 2, 1, 2]),
      features["bool_feature"]: shape([1, 0, 1, 1])
  }

  if has_catset:
    ragged_constant = tf.ragged.constant if is_tf2 else tf.ragged.constant_value

    feature_values[features["d"]] = ragged_constant(
        [["x"], ["y"], ["y", "z"], [""]], dtype=tf.string)

    feature_values[features["e"]] = ragged_constant(
        [[11, 12], [], [14, 15, 16], [-1]], dtype=tf.int32)

  return feature_values


def expected_toy_predictions_rf_weighted(add_boolean_features=False):
  """Expected prediction values."""
  if add_boolean_features:
    probabilities = [[0.0, 0.8, 0.2], [0.5, 0.5, 0.0], [0.1, 0.8, 0.1],
                     [0.8, 0.1, 0.1]]
    classes = [b"v1", b"v2", b"v3"]
  else:
    probabilities = [[0.0, 0.5, 0.5], [0.5, 0.5, 0.0], [0.1, 0.8, 0.1],
                     [0.8, 0.1, 0.1]]
    classes = [b"v1", b"v2", b"v3"]

  return probabilities, classes


def expected_toy_predictions_rf_wta(add_boolean_features=False,
                                    has_catset=False):
  """Expected prediction values."""
  del add_boolean_features
  del has_catset
  probabilities = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]]
  classes = [b"v1", b"v2", b"v3"]
  return probabilities, classes


def expected_toy_predictions_gbdt_binary():
  """Expected prediction values."""
  probabilities = [[1.67e-05, 0.9999833], [1.67e-05, 0.9999833],
                   [0.0474259, 0.9525741], [0.0474259, 0.9525741]]
  classes = [b"v1", b"v2"]
  return probabilities, classes


def expected_toy_predictions_gbdt_multiclass():
  """Expected prediction values."""
  probabilities = [[0.0003345212, 0.0024717960, 0.9971936828],
                   [0.0003345212, 0.0024717960, 0.9971936828],
                   [0.01587624, 0.11731043, 0.86681333],
                   [0.01587624, 0.11731043, 0.86681333]]
  classes = [b"v1", b"v2", b"v3"]
  return probabilities, classes
