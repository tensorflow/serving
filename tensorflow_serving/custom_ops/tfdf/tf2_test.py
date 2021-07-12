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

import tensorflow as tf  # TensorFlow V2

from tensorflow_decision_forests.tensorflow.ops.inference import api as inference
from tensorflow_decision_forests.tensorflow.ops.inference import test_utils
from absl.testing import parameterized
from absl import logging


class TfOpTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("base", False, False),
      ("boolean", True, False),
      ("catset", False, True),
  )
  def test_toy_rf_classification_winner_takes_all(self, add_boolean_features,
                                                  has_catset):

    # Create toy model.
    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
    test_utils.build_toy_random_forest(
        model_path,
        winner_take_all_inference=True,
        add_boolean_features=add_boolean_features,
        has_catset=has_catset)
    features = test_utils.build_toy_input_feature_values(
        features=None, has_catset=has_catset)

    # Prepare model.
    model = inference.Model(model_path)

    @tf.function
    def init_model():
      tf.print("Loading model")
      model.init_op()

    @tf.function
    def apply_model(features):
      tf.print("Running model")
      return model.apply(features)

    init_model()

    predictions = apply_model(features)
    print("Predictions: %s", predictions)

    logging.info("dense_predictions_values: %s", predictions.dense_predictions)
    logging.info("dense_col_representation_values: %s",
                 predictions.dense_col_representation)

    expected_proba, expected_classes = test_utils.expected_toy_predictions_rf_wta(
        add_boolean_features=add_boolean_features, has_catset=has_catset)
    self.assertAllEqual(predictions.dense_col_representation, expected_classes)
    self.assertAllClose(predictions.dense_predictions, expected_proba)

  @parameterized.named_parameters(
      ("base", False, False),
      ("boolean", True, False),
      ("catset", False, True),
  )
  def test_toy_rf_classification_winner_takes_all_v2(self, add_boolean_features,
                                                     has_catset):

    # Create toy model.
    model_path = os.path.join(
        tempfile.mkdtemp(dir=self.get_temp_dir()), "test_basic_rf_wta")
    test_utils.build_toy_random_forest(
        model_path,
        winner_take_all_inference=True,
        add_boolean_features=add_boolean_features,
        has_catset=has_catset)
    features = test_utils.build_toy_input_feature_values(
        features=None, has_catset=has_catset)

    # Prepare model.
    tf.print("Loading model")
    model = inference.ModelV2(model_path)

    @tf.function
    def apply_non_eager(features):
      return model.apply(features)

    predictions_non_eager = apply_non_eager(features)
    predictions_eager = model.apply(features)

    def check_predictions(predictions):
      print("Predictions: %s", predictions)

      logging.info("dense_predictions_values: %s",
                   predictions.dense_predictions)
      logging.info("dense_col_representation_values: %s",
                   predictions.dense_col_representation)

      (expected_proba,
       expected_classes) = test_utils.expected_toy_predictions_rf_wta(
           add_boolean_features=add_boolean_features, has_catset=has_catset)
      self.assertAllEqual(predictions.dense_col_representation,
                          expected_classes)
      self.assertAllClose(predictions.dense_predictions, expected_proba)

    check_predictions(predictions_non_eager)
    check_predictions(predictions_eager)


if __name__ == "__main__":
  tf.test.main()
