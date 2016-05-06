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

#!/usr/grte/v4/bin/python2.7

"""Send JPEG image to inception_inference server for classification.
"""

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.example import inception_inference_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'inception_inference service host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 5


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  # Send request
  with open(FLAGS.image, 'rb') as f:
    # See inception_inference.proto for gRPC request/response details.
    data = f.read()
    request = inception_inference_pb2.InceptionRequest()
    request.jpeg_encoded = data
    result = stub.Classify(request, 10.0)  # 10 secs timeout
    print result


if __name__ == '__main__':
  tf.app.run()
