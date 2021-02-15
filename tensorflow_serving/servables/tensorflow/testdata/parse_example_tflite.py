## Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Exports a simple graph with parse example with a string feature.

Exports a TFLite flatbufer to `/tmp/parse_example_tflite/model.tflite`.

This graph simply parses an example with a simple feature_spec
  {"x": tf.float32[1], "y": tf.string[1]}

\\(
  out = tf.parse_example(x, feature_spec)
  return out["x"], out["y"]
\\)

Output from this program is typically used to exercise TFLite ParseExample
execution.

To create a model:
  bazel run -c opt parse_example_tflite_with_string
"""
import argparse
import sys

# This is a placeholder for a Google-internal import.
import tensorflow.compat.v1 as tf

from tensorflow.lite.tools.signature import signature_def_utils

FLAGS = None


def _get_feature_spec():
  """Returns feature spec for parsing tensorflow.Example."""
  # tensorflow.Example contains two features "x" and "y", where y is a string.
  return {
      "x": tf.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0]),
      "y": tf.FixedLenFeature([1], dtype=tf.string, default_value=["missing"])
  }


def _build_predict_signature(input_tensor, output_tensor_x, output_tensor_y):
  """Helper function for building a predict SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {"input": input_tensor_info}

  output_tensor_info_x = tf.saved_model.utils.build_tensor_info(output_tensor_x)
  output_tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor_y)
  signature_outputs = {"ParseExample/ParseExampleV2": output_tensor_info_x,
                       "ParseExample/ParseExampleV2:1": output_tensor_info_y}
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


def _generate_tflite_for_parse_example_with_string(export_dir):
  """Generates TFLite flatbuffer for parse example with string.

  Args:
    export_dir: The directory to which the flatbuffer should be written.
  """
  with tf.Session(
      graph=tf.Graph(),
      config=tf.ConfigProto(log_device_placement=True)) as sess:
    serialized_tf_example = tf.placeholder(
        tf.string, name="input", shape=[None])
    tf_example = tf.parse_example(serialized_tf_example,
                                  _get_feature_spec())
    converter = tf.lite.TFLiteConverter.from_session(
        sess, [serialized_tf_example], [tf_example["x"], tf_example["y"]])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    predict_signature_def = _build_predict_signature(serialized_tf_example,
                                                     tf_example["x"],
                                                     tf_example["y"])
    k = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tflite_model = signature_def_utils.set_signature_defs(
        tflite_model, {k: predict_signature_def})
    open(export_dir + "/model.tflite", "wb").write(tflite_model)


def main(_):
  _generate_tflite_for_parse_example_with_string(FLAGS.output_dir)
  print("TFLite model generated at: %(dir)s" % {
      "dir": FLAGS.output_dir
  })


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/tmp/parse_example_tflite",
      help="Directory where to output model in TensorFlow Lite format.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
