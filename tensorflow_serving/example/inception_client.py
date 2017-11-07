#!/usr/bin/env python2.7

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import base64
import os

class Processor:
  __stub = None
  __vocab = None
  __wrap = None

  def __init__(self, server, dataDir):
    host, port = server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    self.__stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  def Process(self, modelName, inputList):
    image = base64.decodestring(inputList[0])
    # Send request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = modelName
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image, shape=[1]))
    result = self.__stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)
    myresult = str(result)
    return myresult


if __name__ == '__main__':
  tf.app.run()

