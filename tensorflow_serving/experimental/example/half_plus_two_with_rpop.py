# Copyright 2020 Google Inc. All Rights Reserved.
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
r"""Exports an example inference graph with RemotePredictOp.

Exports a TensorFlow graph to `/tmp/half_plus_two_with_rpop` based on the
`SavedModel` format.

This graph calculates,

\\(
  y = a*x + b
\\)

where `a` is variable with `a=0.5`, and `b` is the first output tensors from
RemotePredictOp.
"""

import tensorflow.compat.v1 as tf

from tensorflow_serving.experimental.tensorflow.ops.remote_predict.python.ops import remote_predict_ops

tf.app.flags.DEFINE_string("output_dir", "/tmp/half_plus_two_with_rpop/1/",
                           "Savedmodel export path")
tf.app.flags.DEFINE_string("target_address", "",
                           "Target address for the RemotePredictOp.")
tf.app.flags.DEFINE_string("remote_model_name", "half_plus_two",
                           "Model name for the RemotePredictOp.")
tf.app.flags.DEFINE_integer("remote_model_version", -1,
                            "Model version for the RemotePredictOp.")

FLAGS = tf.app.flags.FLAGS


def _generate_saved_model_for_half_plus_two(export_dir, target_address,
                                            remote_model_name):
  """Generates SavedModel for half plus two with RemotePredictOp.

  Args:
    export_dir: The directory to which the SavedModel should be written.
    target_address: The target_address for RemotePredictOp.
    remote_model_name: The model name for RemotePredictOp.
  """
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  device_name = "/cpu:0"

  with tf.Session(
      graph=tf.Graph(),
      config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.device(device_name):
      a = tf.Variable(0.5, name="a")

      model_name = remote_model_name
      input_tensor_aliases = tf.constant(["x"])
      input_tensors = [tf.constant(2.0, tf.float32)]

      output_tensor_aliases = tf.constant(["y"])
      output_types = [tf.float32]

      results = remote_predict_ops.run_returning_status(
          input_tensor_aliases,
          input_tensors,
          output_tensor_aliases,
          target_address=target_address,
          model_name=model_name,
          model_version=FLAGS.remote_model_version,
          output_types=output_types)
      b = results.output_tensors[0]

      # Create a placeholder for serialized tensorflow.Example messages to be
      # fed.
      serialized_tf_example = tf.placeholder(tf.string, name="tf_example")

      # Parse the tensorflow.Example looking for a feature named "x" with a
      # single floating point value.
      feature_configs = {"x": tf.FixedLenFeature([1], dtype=tf.float32)}
      # parse_example only works on CPU
      with tf.device("/cpu:0"):
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
      # Use tf.identity() to assign name
      x = tf.identity(tf_example["x"], name="x")
      y = tf.add(tf.multiply(a, x), b, name="y")

    # Set up the signature for Predict with input and output tensor
    # specification.
    predict_input_tensor = tf.saved_model.utils.build_tensor_info(x)
    predict_signature_inputs = {"x": predict_input_tensor}

    predict_output_tensor = tf.saved_model.utils.build_tensor_info(y)
    predict_signature_outputs = {"y": predict_output_tensor}
    predict_signature_def = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=predict_signature_inputs,
            outputs=predict_signature_outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    signature_def_map = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature_def
    }
    # Initialize all variables and then save the SavedModel.
    sess.run(tf.global_variables_initializer())

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        main_op=tf.compat.v1.tables_initializer(),
        strip_default_attrs=True)
  builder.save(False)


def main(_):
  _generate_saved_model_for_half_plus_two(FLAGS.output_dir,
                                          FLAGS.target_address,
                                          FLAGS.remote_model_name)
  print(
      "SavedModel generated at: %(dir)s with target_address: %(target_address)s"
      ", remote_model_name: %(remote_model_name)s. " % {
          "dir": FLAGS.output_dir,
          "target_address": FLAGS.target_address,
          "remote_model_name": FLAGS.remote_model_name
      })


if __name__ == "__main__":
  tf.app.run()
