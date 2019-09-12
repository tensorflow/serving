# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Creates the tf_serving_warmup_requests file to warm up a ResNet SavedModel.

   1. Invoke this script passing in the saved_model directory (including version
        folder, the folder containing saved_model.pb) as an argument.
   2. Restart tensorflow_model_server.

   If unsure of the model directory, look for the output:
   'No warmup data file found at' in the tensorflow_model_server
   startup log

   After the script is run, and tensorflow_model_server is restarted, to verify
   it is working look for the output:
   'Starting to read warmup data for model at' in the tensorflow_model_server
   startup log

   Usage example:
     python resnet_warmup.py saved_model_dir
"""

from __future__ import print_function

import os
import sys
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


# IMAGE_URLS are the locations of the images we use to warmup the model
IMAGE_URLS = ['https://tensorflow.org/images/blogs/serving/cat.jpg',
              # pylint: disable=g-line-too-long
              'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg',
              # pylint: enable=g-line-too-long
             ]


def main():
  if len(sys.argv) != 2 or sys.argv[-1].startswith('-'):
    print('Usage: resnet_warmup.py saved_model_dir')
    sys.exit(-1)

  model_dir = sys.argv[-1]
  if not os.path.isdir(model_dir):
    print('The saved model directory: %s does not exist. '
          'Specify the path of an existing model.' % model_dir)
    sys.exit(-1)

  # Create the assets.extra directory
  assets_dir = os.path.join(model_dir, 'assets.extra')
  if not os.path.exists(assets_dir):
    os.mkdir(assets_dir)

  warmup_file = os.path.join(assets_dir, 'tf_serving_warmup_requests')
  with tf.io.TFRecordWriter(warmup_file) as writer:
    for image in IMAGE_URLS:
      # Download the image
      dl_request = requests.get(image, stream=True)
      dl_request.raise_for_status()
      data = dl_request.content

      # Create the inference request
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'resnet'
      request.model_spec.signature_name = 'serving_default'
      request.inputs['image_bytes'].CopyFrom(
          tf.contrib.util.make_tensor_proto(data, shape=[1]))

      log = prediction_log_pb2.PredictionLog(
          predict_log=prediction_log_pb2.PredictLog(request=request))
      writer.write(log.SerializeToString())

  print('Created the file \'%s\', restart tensorflow_model_server to warmup '
        'the ResNet SavedModel.' % warmup_file)

if __name__ == '__main__':
  main()
