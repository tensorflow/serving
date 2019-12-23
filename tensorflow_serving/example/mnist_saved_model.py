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

#! /usr/bin/env python
r"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.

Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    export_dir
"""

from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

import mnist_input_data

tf.compat.v1.app.flags.DEFINE_integer('training_iteration', 1000,
                                      'number of training iterations.')
tf.compat.v1.app.flags.DEFINE_integer('model_version', 1,
                                      'version number of the model.')
tf.compat.v1.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.disable_eager_execution()


def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_saved_model.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)

  # Train model
  print('Training model...')
  mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
  sess = tf.compat.v1.InteractiveSession()
  serialized_tf_example = tf.compat.v1.placeholder(tf.string, name='tf_example')
  feature_configs = {
      'x': tf.io.FixedLenFeature(shape=[784], dtype=tf.float32),
  }
  tf_example = tf.io.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.compat.v1.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.compat.v1.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  cross_entropy = -tf.math.reduce_sum(y_ * tf.math.log(y))
  train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(
      cross_entropy)
  values, indices = tf.nn.top_k(y, 10)
  table = lookup_ops.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in range(10)]))
  prediction_classes = table.lookup(tf.dtypes.cast(indices, tf.int64))
  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, 'float'))
  print('training accuracy %g' % sess.run(
      accuracy, feed_dict={
          x: mnist.test.images,
          y_: mnist.test.labels
      }))
  print('Done training!')

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
  export_path_base = sys.argv[-1]
  export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
  print('Exporting trained model to', export_path)
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  classification_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(
      serialized_tf_example)
  classification_outputs_classes = tf.compat.v1.saved_model.utils.build_tensor_info(
      prediction_classes)
  classification_outputs_scores = tf.compat.v1.saved_model.utils.build_tensor_info(
      values)

  classification_signature = (
      tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.compat.v1.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classification_inputs
          },
          outputs={
              tf.compat.v1.saved_model.signature_constants
              .CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
              tf.compat.v1.saved_model.signature_constants
              .CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
          },
          method_name=tf.compat.v1.saved_model.signature_constants
          .CLASSIFY_METHOD_NAME))

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_x},
          outputs={'scores': tensor_info_y},
          method_name=tf.compat.v1.saved_model.signature_constants
          .PREDICT_METHOD_NAME))

  builder.add_meta_graph_and_variables(
      sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          tf.compat.v1.saved_model.signature_constants
          .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      main_op=tf.compat.v1.tables_initializer(),
      strip_default_attrs=True)

  builder.save()

  print('Done exporting!')


if __name__ == '__main__':
  tf.compat.v1.app.run()
