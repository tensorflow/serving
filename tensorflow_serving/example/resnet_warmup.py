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
"""Creates the tf_serving_warmup_requests file to warm up a server.
   
   1. Invoke this script to generate the tf_serving_warmup_requests file.
   2. Create a directory 'assets.extra' in the folder of the saved model.
   3. copy the file tf_serving_warmup_requests into that folder.
   4. Restart tensorflow_model_server.

   If unsure where the file goes, look for the output:
   'No warmup data file found at' in the tensorflow_model_server
   startup log

   When copied to the proper location look for the output:
   'Starting to read warmup data for model at' in the tensorflow_model_server
   startup log

   Usage example:
     python resnet_warmup.py
"""
import requests
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

def main():
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()
  data = dl_request.content

  with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['image_bytes'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))

    log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
    writer.write(log.SerializeToString())

  print("Copy the generated tf_serving_warmup_requests to the " \
        "assets.extra directory of the model")

if __name__ == '__main__':
  main()
