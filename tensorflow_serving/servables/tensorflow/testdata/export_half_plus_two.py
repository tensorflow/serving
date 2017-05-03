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
"""Exports a toy TensorFlow model.

Exports a TensorFlow model to /tmp/half_plus_two/.

This graph calculates,
  y = a*x + b
where a and b are variables with a=0.5 and b=2.
"""

# This is a placeholder for a Google-internal import.

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter


def Export():
  export_path = "/tmp/half_plus_two"
  with tf.Session() as sess:
    # Make model parameters a&b variables instead of constants to
    # exercise the variable reloading mechanisms.
    a = tf.Variable(0.5)
    b = tf.Variable(2.0)

    # Calculate, y = a*x + b
    # here we use a placeholder 'x' which is fed at inference time.
    x = tf.placeholder(tf.float32)
    y = tf.add(tf.multiply(a, x), b)

    # Run an export.
    tf.global_variables_initializer().run()
    export = exporter.Exporter(tf.train.Saver())
    export.init(named_graph_signatures={
        "inputs": exporter.generic_signature({"x": x}),
        "outputs": exporter.generic_signature({"y": y}),
        "regress": exporter.regression_signature(x, y)
    })
    export.export(export_path, tf.constant(123), sess)


def main(_):
  Export()


if __name__ == "__main__":
  tf.app.run()
