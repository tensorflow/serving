# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Exports a counter model.

It contains 4 signatures: get_counter incr_counter, incr_counter_by, and
reset_counter, to test Predict service.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This is a placeholder for a Google-internal import.
import tensorflow as tf


def save_model(sess, signature_def_map, output_dir):
  """Saves the model with given signature def map."""
  builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map=signature_def_map)
  builder.save()


def build_signature_def_from_tensors(inputs, outputs, method_name):
  """Builds signature def with inputs, outputs, and method_name."""
  return tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in inputs.items()
      },
      outputs={
          key: tf.saved_model.utils.build_tensor_info(tensor)
          for key, tensor in outputs.items()
      },
      method_name=method_name)


def export_model(output_dir):
  """Exports the counter model.

  Create three signatures: incr_counter, incr_counter_by, reset_counter.

  *Notes*: These signatures are stateful and over-simplied only to demonstrate
  Predict calls with only inputs or outputs. State is not supported in
  TensorFlow Serving on most scalable or production hosting environments.

  Args:
    output_dir: string, output directory for the model.
  """
  tf.logging.info("Exporting the counter model to %s.", output_dir)
  method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME

  graph = tf.Graph()
  with graph.as_default(), tf.Session() as sess:
    counter = tf.Variable(0.0, dtype=tf.float32, name="counter")

    with tf.name_scope("incr_counter_op", values=[counter]):
      incr_counter = counter.assign_add(1.0)

    delta = tf.placeholder(dtype=tf.float32, name="delta")
    with tf.name_scope("incr_counter_by_op", values=[counter, delta]):
      incr_counter_by = counter.assign_add(delta)

    with tf.name_scope("reset_counter_op", values=[counter]):
      reset_counter = counter.assign(0.0)

    sess.run(tf.global_variables_initializer())

    signature_def_map = {
        "get_counter":
            build_signature_def_from_tensors({}, {"output": counter},
                                             method_name),
        "incr_counter":
            build_signature_def_from_tensors({}, {"output": incr_counter},
                                             method_name),
        "incr_counter_by":
            build_signature_def_from_tensors({
                "delta": delta
            }, {"output": incr_counter_by}, method_name),
        "reset_counter":
            build_signature_def_from_tensors({}, {"output": reset_counter},
                                             method_name)
    }
    save_model(sess, signature_def_map, output_dir)


def main(unused_argv):
  export_model("/tmp/saved_model_counter/00000123")


if __name__ == "__main__":
  tf.compat.v1.app.run()
