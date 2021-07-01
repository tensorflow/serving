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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow.compat.v1 as tf

from tensorflow_decision_forests.tensorflow.ops.inference import api as inference
from tensorflow_decision_forests.tensorflow.ops.inference import test_utils
from absl.testing import parameterized
from absl import flags
from absl import logging


def data_root_path() -> str:
  return ""


def test_data_path() -> str:
  return os.path.join(data_root_path(),
                      "external/ydf/yggdrasil_decision_forests/test_data")


class TfOpTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("base", False, False),
      ("boolean", True, False),
      ("catset", False, True),
  )
  def test_toy_rf_classification_winner_takes_all(self, add_boolean_features,
                                                  has_catset):

    with tf.Graph().as_default():

      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
      test_utils.build_toy_random_forest(
          model_path,
          winner_take_all_inference=True,
          add_boolean_features=add_boolean_features,
          has_catset=has_catset)
      features = test_utils.build_toy_input_features(has_catset=has_catset)

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        dense_predictions_values, dense_col_representation_values = sess.run(
            [
                predictions.dense_predictions,
                predictions.dense_col_representation
            ],
            test_utils.build_toy_input_feature_values(
                features, has_catset=has_catset))
        logging.info("dense_predictions_values: %s", dense_predictions_values)
        logging.info("dense_col_representation_values: %s",
                     dense_col_representation_values)

        expected_proba, expected_classes = test_utils.expected_toy_predictions_rf_wta(
            add_boolean_features=add_boolean_features, has_catset=has_catset)
        self.assertAllEqual(dense_col_representation_values, expected_classes)
        self.assertAllClose(dense_predictions_values, expected_proba)

  @parameterized.named_parameters(("base", False), ("boolean", True))
  def test_toy_rf_classification_weighted(self, add_boolean_features):

    with tf.Graph().as_default():

      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_weighted")
      test_utils.build_toy_random_forest(
          model_path,
          winner_take_all_inference=False,
          add_boolean_features=add_boolean_features)
      features = test_utils.build_toy_input_features()

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        dense_predictions_values, dense_col_representation_values = sess.run([
            predictions.dense_predictions, predictions.dense_col_representation
        ], test_utils.build_toy_input_feature_values(features))
        logging.info("dense_predictions_values: %s", dense_predictions_values)
        logging.info("dense_col_representation_values: %s",
                     dense_col_representation_values)

        expected_proba, expected_classes = test_utils.expected_toy_predictions_rf_weighted(
            add_boolean_features=add_boolean_features)
        self.assertAllEqual(dense_col_representation_values, expected_classes)
        self.assertAllClose(dense_predictions_values, expected_proba)

  @parameterized.named_parameters(("base", False), ("boolean", True))
  def test_toy_rf_classification_weighted_rank2(self, add_boolean_features):

    with tf.Graph().as_default():

      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_weightd")
      test_utils.build_toy_random_forest(
          model_path,
          winner_take_all_inference=False,
          add_boolean_features=add_boolean_features)
      features = test_utils.build_toy_input_features(use_rank_two=True)

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        dense_predictions_values, dense_col_representation_values = sess.run([
            predictions.dense_predictions, predictions.dense_col_representation
        ], test_utils.build_toy_input_feature_values(
            features, use_rank_two=True))
        logging.info("dense_predictions_values: %s", dense_predictions_values)
        logging.info("dense_col_representation_values: %s",
                     dense_col_representation_values)

        expected_proba, expected_classes = test_utils.expected_toy_predictions_rf_weighted(
            add_boolean_features=add_boolean_features)
        self.assertAllEqual(dense_col_representation_values, expected_classes)
        self.assertAllClose(dense_predictions_values, expected_proba)

  def test_toy_gbdt_binary_classification(self):

    with tf.Graph().as_default():
      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_gbdt_binary")
      test_utils.build_toy_gbdt(model_path, num_classes=2)
      features = test_utils.build_toy_input_features()

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        dense_predictions_values, dense_col_representation_values = sess.run([
            predictions.dense_predictions, predictions.dense_col_representation
        ], test_utils.build_toy_input_feature_values(features))
        logging.info("dense_predictions_values: %s", dense_predictions_values)
        logging.info("dense_col_representation_values: %s",
                     dense_col_representation_values)

        expected_proba, expected_classes = test_utils.expected_toy_predictions_gbdt_binary(
        )
        self.assertAllEqual(dense_col_representation_values, expected_classes)
        self.assertAllClose(dense_predictions_values, expected_proba)

  def test_toy_gbdt_multiclass_classification(self):

    with tf.Graph().as_default():
      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()),
          "test_basic_gbdt_multiclass")
      test_utils.build_toy_gbdt(model_path, num_classes=3)
      features = test_utils.build_toy_input_features()

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        dense_predictions_values, dense_col_representation_values = sess.run([
            predictions.dense_predictions, predictions.dense_col_representation
        ], test_utils.build_toy_input_feature_values(features))
        logging.info("dense_predictions_values: %s", dense_predictions_values)
        logging.info("dense_col_representation_values: %s",
                     dense_col_representation_values)

        expected_proba, expected_classes = test_utils.expected_toy_predictions_gbdt_multiclass(
        )
        self.assertAllEqual(dense_col_representation_values, expected_classes)
        self.assertAllClose(dense_predictions_values, expected_proba)

  def test_real_rf(self):
    """Loads a real Random Forest model."""

    with tf.Graph().as_default():

      model_path = os.path.join(test_data_path(), "model",
                                "adult_binary_class_rf")

      features = {
          "age": tf.placeholder(tf.float32, [None]),
          "workclass": tf.placeholder(tf.string, [None]),
          "fnlwgt": tf.placeholder(tf.float32, [None]),
          "education": tf.placeholder(tf.string, [None]),
          "education_num": tf.placeholder(tf.int32, [None]),
          "marital_status": tf.placeholder(tf.string, [None]),
          "occupation": tf.placeholder(tf.string, [None]),
          "relationship": tf.placeholder(tf.string, [None]),
          "race": tf.placeholder(tf.string, [None]),
          "sex": tf.placeholder(tf.string, [None]),
          "capital_gain": tf.placeholder(tf.float32, [None]),
          "capital_loss": tf.placeholder(tf.float32, [None]),
          "hours_per_week": tf.placeholder(tf.float32, [None]),
          "native_country": tf.placeholder(tf.string, [None]),
      }

      model = inference.Model(model_path)
      _ = model.apply(features)

      with self.session() as sess:
        sess.run(model.init_op())

  def test_real_gbdt(self):
    """Loads a real GBDT model."""

    with tf.Graph().as_default():

      model_path = os.path.join(test_data_path(), "model",
                                "adult_binary_class_gbdt")

      features = {
          "age": tf.placeholder(tf.float32, [None]),
          "workclass": tf.placeholder(tf.string, [None]),
          "fnlwgt": tf.placeholder(tf.float32, [None]),
          "education": tf.placeholder(tf.string, [None]),
          "education_num": tf.placeholder(tf.int32, [None]),
          "marital_status": tf.placeholder(tf.string, [None]),
          "occupation": tf.placeholder(tf.string, [None]),
          "relationship": tf.placeholder(tf.string, [None]),
          "race": tf.placeholder(tf.string, [None]),
          "sex": tf.placeholder(tf.string, [None]),
          "capital_gain": tf.placeholder(tf.float32, [None]),
          "capital_loss": tf.placeholder(tf.float32, [None]),
          "hours_per_week": tf.placeholder(tf.float32, [None]),
          "native_country": tf.placeholder(tf.string, [None]),
      }

      model = inference.Model(model_path)
      _ = model.apply(features)

      with self.session() as sess:
        sess.run(model.init_op())

  def test_crash_op(self):
    with tf.Graph().as_default():

      # Create toy model.
      model_path = os.path.join(
          tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
      test_utils.build_toy_random_forest(
          model_path, winner_take_all_inference=True, has_catset=True)
      features = test_utils.build_toy_input_features(has_catset=True)

      # Prepare model.
      model = inference.Model(model_path)
      predictions = model.apply(features)

      # Run model on toy dataset.
      with self.session() as sess:
        sess.run(model.init_op())

        def good_feature_values():
          return test_utils.build_toy_input_feature_values(
              features, has_catset=True)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          feature_values = good_feature_values()
          feature_values = {features["a"]: feature_values[features["a"]]}
          sess.run([
              predictions.dense_predictions,
              predictions.dense_col_representation
          ], feature_values)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          feature_values = good_feature_values()
          feature_values[features["a"]] = [2, 2, 0, 0, 6]
          sess.run([
              predictions.dense_predictions,
              predictions.dense_col_representation
          ], feature_values)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          feature_values = good_feature_values()
          feature_values[features["b"]] = ["x", "z", "x", "z", "x"]
          sess.run([
              predictions.dense_predictions,
              predictions.dense_col_representation
          ], feature_values)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          feature_values = good_feature_values()
          feature_values[features["e"]] = tf.ragged.constant_value(
              [[], [], [], [], []], dtype=tf.int32)
          sess.run([
              predictions.dense_predictions,
              predictions.dense_col_representation
          ], feature_values)

        with self.assertRaises(tf.errors.InvalidArgumentError):
          feature_values = good_feature_values()
          feature_values[features["e"]] = tf.ragged.constant_value(
              [[], []], dtype=tf.int32)
          sess.run([
              predictions.dense_predictions,
              predictions.dense_col_representation
          ], feature_values)


if __name__ == "__main__":
  tf.test.main()
