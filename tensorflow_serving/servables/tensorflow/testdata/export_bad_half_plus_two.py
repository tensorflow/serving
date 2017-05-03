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
"""Exports a toy TensorFlow model without signatures.

Exports half_plus_two TensorFlow model to /tmp/bad_half_plus_two/ without
signatures. This is used to test the fault-tolerance of tensorflow_model_server.
"""

import os

# This is a placeholder for a Google-internal import.

import tensorflow as tf


def Export():
  export_path = "/tmp/bad_half_plus_two/00000123"
  with tf.Session() as sess:
    # Make model parameters a&b variables instead of constants to
    # exercise the variable reloading mechanisms.
    a = tf.Variable(0.5)
    b = tf.Variable(2.0)

    # Calculate, y = a*x + b
    # here we use a placeholder 'x' which is fed at inference time.
    x = tf.placeholder(tf.float32)
    y = tf.add(tf.multiply(a, x), b)

    # Export the model without signatures.
    # Note that the model is intentionally exported without using exporter,
    # but using the same format. This is to avoid exporter creating default
    # empty signatures upon export.
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.export_meta_graph(
        filename=os.path.join(export_path, "export.meta"))
    saver.save(sess,
               os.path.join(export_path, "export"),
               write_meta_graph=False)


def main(_):
  Export()


if __name__ == "__main__":
  tf.app.run()
