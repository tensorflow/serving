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
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

import io

# This is a placeholder for a Google-internal import.

import grpc
import numpy as np
from PIL import Image
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('image', '',
                                     'path to image in JPEG format')
FLAGS = tf.compat.v1.app.flags.FLAGS

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False


def main(_):
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      data = f.read()
  else:
    # Download the image since we weren't given one
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    data = dl_request.content

  if not MODEL_ACCEPT_JPG:
    data = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    data = np.array(data) / 255.0
    data = np.expand_dims(data, 0)
    data = data.astype(np.float32)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['input_1'].CopyFrom(
      tf.make_tensor_proto(data))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  result = result.outputs['activation_49'].float_val
  print('Prediction class: {}'.format(np.argmax(result)))


if __name__ == '__main__':
  tf.compat.v1.app.run()
