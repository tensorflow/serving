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
r"""Remote Predict Op client example.

Example client code which calls the Remote Predict Op directly.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import tensorflow.compat.v1 as tf

from tensorflow_serving.experimental.tensorflow.ops.remote_predict.python.ops import remote_predict_ops

tf.app.flags.DEFINE_string("input_tensor_aliases", "x",
                           "Aliases of input tensors")
tf.app.flags.DEFINE_float("input_value", 1.0, "input value")
tf.app.flags.DEFINE_string("output_tensor_aliases", "y",
                           "Aliases of output tensors")

tf.app.flags.DEFINE_string("target_address", "localhost:8500",
                           "PredictionService address host:port")
tf.app.flags.DEFINE_string("model_name", "half_plus_two", "Name of the model")
tf.app.flags.DEFINE_integer("model_version", -1, "Version of the model")
tf.app.flags.DEFINE_boolean("fail_op_on_rpc_error", True, "Failure handling")
tf.app.flags.DEFINE_integer("rpc_deadline_millis", 30000,
                            "rpc deadline in milliseconds")

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
  print("Call remote_predict_op")
  results = remote_predict_ops.run(
      [FLAGS.input_tensor_aliases],
      [tf.constant(FLAGS.input_value, dtype=tf.float32)],
      [FLAGS.output_tensor_aliases],
      target_address=FLAGS.target_address,
      model_name=FLAGS.model_name,
      model_version=FLAGS.model_version,
      fail_op_on_rpc_error=FLAGS.fail_op_on_rpc_error,
      max_rpc_deadline_millis=FLAGS.rpc_deadline_millis,
      output_types=[tf.float32])
  print("Done remote_predict_op")
  print("Returned Result:", results.output_tensors[0].numpy())


if __name__ == "__main__":
  tf.app.run()
