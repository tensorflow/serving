## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
r"""Exports an example linear regression inference graph.

Exports a TensorFlow graph to `/tmp/saved_model/half_plus_two/` based on the
`SavedModel` format.

This graph calculates,

\\(
  y = a*x + b
\\)

and/or, independently,

\\(
  y2 = a*x2 + c
\\)

where `a`, `b` and `c` are variables with `a=0.5` and `b=2` and `c=3`.

Output from this program is typically used to exercise SavedModel load and
execution code.

To create a CPU model:
  bazel run -c opt saved_model_half_plus_two -- --device=cpu

To create a CPU model with Intel MKL-DNN optimizations:
  bazel run -c opt saved_model_half_plus_two -- --device=mkl

To create GPU model:
  bazel run --config=cuda -c opt saved_model_half_plus_two -- \
  --device=gpu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

# This is a placeholder for a Google-internal import.
import tensorflow.compat.v1 as tf

from tensorflow.lite.tools.signature import signature_def_utils
from tensorflow.python.lib.io import file_io

FLAGS = None


def _get_feature_spec():
  """Returns feature spec for parsing tensorflow.Example."""
  # tensorflow.Example contains two features "x" and "x2".
  return {
      "x": tf.FixedLenFeature([1], dtype=tf.float32),
      "x2": tf.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0])
  }


def _write_assets(assets_directory, assets_filename):
  """Writes asset files to be used with SavedModel for half plus two.

  Args:
    assets_directory: The directory to which the assets should be written.
    assets_filename: Name of the file to which the asset contents should be
        written.

  Returns:
    The path to which the assets file was written.
  """
  if not file_io.file_exists(assets_directory):
    file_io.recursive_create_dir(assets_directory)

  path = os.path.join(
      tf.compat.as_bytes(assets_directory), tf.compat.as_bytes(assets_filename))
  file_io.write_string_to_file(path, "asset-file-contents")
  return path


def _build_predict_signature(input_tensor, output_tensor):
  """Helper function for building a predict SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {"x": input_tensor_info}

  output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
  signature_outputs = {"y": output_tensor_info}
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


def _build_regression_signature(input_tensor, output_tensor):
  """Helper function for building a regression SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.REGRESS_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.REGRESS_OUTPUTS: output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.REGRESS_METHOD_NAME)


# Possibly extend this to allow passing in 'classes', but for now this is
# sufficient for testing purposes.
def _build_classification_signature(input_tensor, scores_tensor):
  """Helper function for building a classification SignatureDef."""
  input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
  signature_inputs = {
      tf.saved_model.signature_constants.CLASSIFY_INPUTS: input_tensor_info
  }
  output_tensor_info = tf.saved_model.utils.build_tensor_info(scores_tensor)
  signature_outputs = {
      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
          output_tensor_info
  }
  return tf.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)


def _create_asset_file(tf2=False):
  """Helper to create assets file. Returns a tensor for the filename."""
  # Create an assets file that can be saved and restored as part of the
  # SavedModel.
  original_assets_directory = "/tmp/original/export/assets"
  original_assets_filename = "foo.txt"
  original_assets_filepath = _write_assets(original_assets_directory,
                                           original_assets_filename)

  if tf2:
    return tf.saved_model.Asset(original_assets_filepath)

  # Set up the assets collection.
  assets_filepath = tf.constant(original_assets_filepath)
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
  filename_tensor = tf.Variable(
      original_assets_filename,
      name="filename_tensor",
      trainable=False,
      collections=[])
  return filename_tensor.assign(original_assets_filename)


def _write_mlmd(export_dir, mlmd_uuid):
  """Writes an ML Metadata UUID into the assets.extra directory.

  Args:
    export_dir: The export directory for the SavedModel.
    mlmd_uuid: The string to write as the ML Metadata UUID.

  Returns:
    The path to which the MLMD UUID was written.
  """
  assets_extra_directory = os.path.join(export_dir, "assets.extra")
  if not file_io.file_exists(assets_extra_directory):
    file_io.recursive_create_dir(assets_extra_directory)
  path = os.path.join(assets_extra_directory, "mlmd_uuid")
  file_io.write_string_to_file(path, mlmd_uuid)
  return path


class HalfPlusTwoModel(tf.Module):
  """Native TF2 half-plus-two model."""

  def __init__(self):
    self.a = tf.Variable(0.5, name="a")
    self.b = tf.Variable(2.0, name="b")
    self.c = tf.Variable(3.0, name="c")
    self.asset = _create_asset_file(tf2=True)

  def compute(self, x, inc):
    return tf.add(tf.multiply(self.a, x), inc)

  def get_serving_signatures(self):
    return {
        "regress_x_to_y": self.regress_xy,
        "regress_x_to_y2": self.regress_xy2,
        "regress_x2_to_y3": self.regress_x2y3,
        "classify_x_to_y": self.classify_xy,
        "classify_x2_to_y3": self.classify_x2y3,
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.predict,
    }

  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
  def predict(self, x=tf.constant([0], shape=[1], dtype=tf.float32)):
    return {"y": self.compute(x, self.b)}

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              [None], dtype=tf.string, name=tf.saved_model.REGRESS_INPUTS
          )
      ]
  )
  def regress_xy(self, serialized_proto):
    x = tf.parse_example(serialized_proto, _get_feature_spec())["x"]
    return {tf.saved_model.REGRESS_OUTPUTS: self.compute(x, self.b)}

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              [None], dtype=tf.string, name=tf.saved_model.REGRESS_INPUTS
          )
      ]
  )
  def regress_xy2(self, serialized_proto):
    x = tf.parse_example(serialized_proto, _get_feature_spec())["x"]
    return {tf.saved_model.REGRESS_OUTPUTS: self.compute(x, self.c)}

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[1], dtype=tf.float32, name=tf.saved_model.REGRESS_INPUTS
          )
      ]
  )
  def regress_x2y3(self, x2):
    return {tf.saved_model.REGRESS_OUTPUTS: self.compute(x2, self.c)}

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              [None], dtype=tf.string, name=tf.saved_model.CLASSIFY_INPUTS
          )
      ]
  )
  def classify_xy(self, serialized_proto):
    x = tf.parse_example(serialized_proto, _get_feature_spec())["x"]
    return {tf.saved_model.CLASSIFY_OUTPUT_SCORES: self.compute(x, self.b)}

  @tf.function(
      input_signature=[
          tf.TensorSpec(
              shape=[1], dtype=tf.float32, name=tf.saved_model.CLASSIFY_INPUTS
          )
      ]
  )
  def classify_x2y3(self, x2):
    return {tf.saved_model.CLASSIFY_OUTPUT_SCORES: self.compute(x2, self.c)}


def _generate_saved_model_for_half_plus_two(
    export_dir,
    tf2=False,
    as_text=False,
    as_tflite=False,
    as_tflite_with_sigdef=False,
    use_main_op=False,
    include_mlmd=False,
    device_type="cpu",
):
  """Generates SavedModel for half plus two.

  Args:
    export_dir: The directory to which the SavedModel should be written.
    tf2: If True generates a SavedModel using native (non compat) TF2 APIs.
    as_text: Writes the SavedModel protocol buffer in text format to disk.
    as_tflite: Writes the Model in Tensorflow Lite format to disk.
    as_tflite_with_sigdef: Writes the Model with SignatureDefs in Tensorflow
      Lite format to disk.
    use_main_op: Whether to supply a main op during SavedModel build time.
    include_mlmd: Whether to include an MLMD key in the SavedModel.
    device_type: Device to force ops to run on.
  """
  if tf2:
    hp = HalfPlusTwoModel()
    tf.saved_model.save(hp, export_dir, signatures=hp.get_serving_signatures())
    return

  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  device_name = "/cpu:0"
  if device_type == "gpu":
    device_name = "/gpu:0"

  with tf.Session(
      graph=tf.Graph(),
      config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device(device_name):
      # Set up the model parameters as variables to exercise variable loading
      # functionality upon restore.
      a = tf.Variable(0.5, name="a")
      b = tf.Variable(2.0, name="b")
      c = tf.Variable(3.0, name="c")

      # Create a placeholder for serialized tensorflow.Example messages to be
      # fed.
      serialized_tf_example = tf.placeholder(
          tf.string, name="tf_example", shape=[None])

      # parse_example only works on CPU
      with tf.device("/cpu:0"):
        tf_example = tf.parse_example(serialized_tf_example,
                                      _get_feature_spec())

      if as_tflite:
        # TFLite v1 converter does not support unknown shape.
        x = tf.ensure_shape(tf_example["x"], (1, 1), name="x")
      else:
        # Use tf.identity() to assign name
        x = tf.identity(tf_example["x"], name="x")

      if as_tflite_with_sigdef:
        # Resulting TFLite model will have input named "tflite_input".
        x = tf.ensure_shape(tf_example["x"], (1, 1), name="tflite_input")

      if device_type == "mkl":
        # Create a small convolution op to trigger MKL
        # The op will return 0s so this won't affect the
        # resulting calculation.
        o1 = tf.keras.layers.Conv2D(1, [1, 1])(tf.zeros((1, 16, 16, 1)))
        y = o1[0, 0, 0, 0] + tf.add(tf.multiply(a, x), b)
      else:
        y = tf.add(tf.multiply(a, x), b)

      y = tf.identity(y, name="y")

      if device_type == "mkl":
        # Create a small convolution op to trigger MKL
        # The op will return 0s so this won't affect the
        # resulting calculation.
        o2 = tf.keras.layers.Conv2D(1, [1, 1])(tf.zeros((1, 16, 16, 1)))
        y2 = o2[0, 0, 0, 0] + tf.add(tf.multiply(a, x), c)
      else:
        y2 = tf.add(tf.multiply(a, x), c)

      y2 = tf.identity(y2, name="y2")

      x2 = tf.identity(tf_example["x2"], name="x2")

      if device_type == "mkl":
        # Create a small convolution op to trigger MKL
        # The op will return 0s so this won't affect the
        # resulting calculation.
        o3 = tf.keras.layers.Conv2D(1, [1, 1])(tf.zeros((1, 16, 16, 1)))
        y3 = o3[0, 0, 0, 0] + tf.add(tf.multiply(a, x2), c)
      else:
        # Add separate constants for x2, to prevent optimizers like TF-TRT from
        # fusing the paths to compute y/y2 and y3 together.
        a2 = tf.Variable(0.5, name="a2")
        c2 = tf.Variable(3.0, name="c2")
        y3 = tf.add(tf.multiply(a2, x2), c2)

      y3 = tf.identity(y3, name="y3")

    assign_filename_op = _create_asset_file()

    predict_signature_def = _build_predict_signature(x, y)

    signature_def_map = {
        "regress_x_to_y":
            _build_regression_signature(serialized_tf_example, y),
        "regress_x_to_y2":
            _build_regression_signature(serialized_tf_example, y2),
        "regress_x2_to_y3":
            _build_regression_signature(x2, y3),
        "classify_x_to_y":
            _build_classification_signature(serialized_tf_example, y),
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            _build_predict_signature(x, y)
    }
    # Initialize all variables and then save the SavedModel.
    sess.run(tf.global_variables_initializer())

    if as_tflite or as_tflite_with_sigdef:
      converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y])
      tflite_model = converter.convert()
      if as_tflite_with_sigdef:
        k = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        tflite_model = signature_def_utils.set_signature_defs(
            tflite_model, {k: predict_signature_def})
      open(export_dir + "/model.tflite", "wb").write(tflite_model)
    else:
      if use_main_op:
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
            main_op=tf.group(tf.saved_model.main_op.main_op(),
                             assign_filename_op))
      else:
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
            main_op=tf.group(assign_filename_op))

  if not as_tflite:
    builder.save(as_text)

  if include_mlmd:
    _write_mlmd(export_dir, "test_mlmd_uuid")


def main(_):
  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir, device_type=FLAGS.device)
  print("SavedModel generated for %(device)s at: %(dir)s" % {
      "device": FLAGS.device,
      "dir": FLAGS.output_dir
  })

  _generate_saved_model_for_half_plus_two(
      "%s_%s" % (FLAGS.output_dir_tf2, FLAGS.device),
      tf2=True,
      device_type=FLAGS.device)
  print(
      "SavedModel TF2 generated for %(device)s at: %(dir)s" % {
          "device": FLAGS.device,
          "dir": "%s_%s" % (FLAGS.output_dir_tf2, FLAGS.device),
      })

  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir_pbtxt, as_text=True, device_type=FLAGS.device)
  print("SavedModel generated for %(device)s at: %(dir)s" % {
      "device": FLAGS.device,
      "dir": FLAGS.output_dir_pbtxt
  })

  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir_main_op, use_main_op=True, device_type=FLAGS.device)
  print("SavedModel generated for %(device)s at: %(dir)s " % {
      "device": FLAGS.device,
      "dir": FLAGS.output_dir_main_op
  })

  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir_tflite, as_tflite=True, device_type=FLAGS.device)
  print("SavedModel in TFLite format generated for %(device)s at: %(dir)s " % {
      "device": FLAGS.device,
      "dir": FLAGS.output_dir_tflite,
  })

  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir_mlmd, include_mlmd=True, device_type=FLAGS.device)
  print("SavedModel with MLMD generated for %(device)s at: %(dir)s " % {
      "device": FLAGS.device,
      "dir": FLAGS.output_dir_mlmd,
  })

  _generate_saved_model_for_half_plus_two(
      FLAGS.output_dir_tflite_with_sigdef, device_type=FLAGS.device,
      as_tflite_with_sigdef=True)
  print("SavedModel in TFLite format with SignatureDef generated for "
        "%(device)s at: %(dir)s " % {
            "device": FLAGS.device,
            "dir": FLAGS.output_dir_tflite_with_sigdef,
        })


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/tmp/saved_model_half_plus_two",
      help="Directory where to output SavedModel.")
  parser.add_argument(
      "--output_dir_tf2",
      type=str,
      default="/tmp/saved_model_half_plus_two_tf2",
      help="Directory where to output TF2 SavedModel.")
  parser.add_argument(
      "--output_dir_pbtxt",
      type=str,
      default="/tmp/saved_model_half_plus_two_pbtxt",
      help="Directory where to output the text format of SavedModel.")
  parser.add_argument(
      "--output_dir_main_op",
      type=str,
      default="/tmp/saved_model_half_plus_two_main_op",
      help="Directory where to output the SavedModel with a main op.")
  parser.add_argument(
      "--output_dir_tflite",
      type=str,
      default="/tmp/saved_model_half_plus_two_tflite",
      help="Directory where to output model in TensorFlow Lite format.")
  parser.add_argument(
      "--output_dir_mlmd",
      type=str,
      default="/tmp/saved_model_half_plus_two_mlmd",
      help="Directory where to output the SavedModel with ML Metadata.")
  parser.add_argument(
      "--output_dir_tflite_with_sigdef",
      type=str,
      default="/tmp/saved_model_half_plus_two_tflite_with_sigdef",
      help=("Directory where to output model with signature def in "
            "TensorFlow Lite format."))
  parser.add_argument(
      "--device",
      type=str,
      default="cpu",
      help="Force model to run on 'cpu', 'mkl', or 'gpu'")
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
