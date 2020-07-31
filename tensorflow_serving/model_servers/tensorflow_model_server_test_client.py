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
"""Manual test client for tensorflow_model_server."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'inception_inference service host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS


def main(_):
  # Prepare request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'default'
  request.inputs['x'].dtype = types_pb2.DT_FLOAT
  request.inputs['x'].float_val.append(2.0)
  request.output_filter.append('y')
  # Send request
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  print(stub.Predict(request, 5.0))  # 5 secs timeout


if __name__ == '__main__':
  tf.compat.v1.app.run()
