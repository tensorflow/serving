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
'Starting to read warmup data for model at' and 'Finished reading warmup data
for model at' in the tensorflow_model_server startup log

Usage example:
python resnet_warmup.py saved_model_dir
"""


import io
import os
import sys

import numpy as np
import requests
import tensorflow as tf
from PIL import Image

from tensorflow_serving.apis import predict_pb2, prediction_log_pb2

# IMAGE_URLS are the locations of the images we use to warmup the model
IMAGE_URLS = ['https://tensorflow.org/images/blogs/serving/cat.jpg',
              # pylint: disable=g-line-too-long
              'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg',
              # pylint: enable=g-line-too-long
             ]

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False


def main():
  if len(sys.argv) != 2 or sys.argv[-1].startswith('-'):
    print('Usage: resnet_warmup.py saved_model_dir')
    sys.exit(-1)

  model_dir = sys.argv[-1]
  if not os.path.isdir(model_dir):
    print(f'The saved model directory: {model_dir} does not exist. '
          'Specify the path of an existing model.')
    sys.exit(-1)

  # Create the assets.extra directory, assuming model_dir is the versioned
  # directory containing the SavedModel
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

      if not MODEL_ACCEPT_JPG:
        data = Image.open(io.BytesIO(dl_request.content))
        # Normalize and batchify the image
        data = np.array(data) / 255.0
        data = np.expand_dims(data, 0)
        data = data.astype(np.float32)

      # Create the inference request
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'resnet'
      request.model_spec.signature_name = 'serving_default'
      request.inputs['input_1'].CopyFrom(
          tf.make_tensor_proto(data))

      log = prediction_log_pb2.PredictionLog(
          predict_log=prediction_log_pb2.PredictLog(request=request))
      writer.write(log.SerializeToString())

  print(f'Created the file \'{warmup_file}\', restart tensorflow_model_server to warmup '
        'the ResNet SavedModel.')

if __name__ == '__main__':
  main()
