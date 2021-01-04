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

This client sample code will send a request to query model
/tmp/half_plus_two_with_rpop which contains Remote Predict Op. The Remote
Predict Op will send a request to query /tmp/half_plus_two. For more details
about /tmp/half_plus_two_with_rpop, please refer to half_plus_two_with_rpop.py.

To run this client example locally, please create a server config file like:
  model_config_list {
    config: {
      name: "half_plus_two"
      base_path: "/tmp/half_plus_two"
      model_platform: "tensorflow"
    }
    config: {
      name: "half_plus_two_with_rpop"
      base_path: "/tmp/half_plus_two_with_rpop"
      model_platform: "tensorflow"
    }
  }
And then run
tensorflow_model_server --port=8500 --model_config_file=/tmp/config_file.txt

"""
from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow.compat.v1 as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'half_plus_two_with_rpop'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['x'].CopyFrom(tf.make_tensor_proto([10.0], shape=[1]))
  result = stub.Predict(request, 30)
  print(result)


if __name__ == '__main__':
  tf.compat.v1.app.run()
