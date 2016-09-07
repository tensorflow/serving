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

#!/usr/bin/env python2.7

"""Tests for tensorflow_model_server."""

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time

# This is a placeholder for a Google-internal import.

from grpc import *
from grpc.beta import implementations
from grpc.framework.interfaces.face import face
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

FLAGS = flags.FLAGS


def PickUnusedPort():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(('localhost', 0))
  _, port = s.getsockname()
  s.close()
  return port


class TensorflowModelServerTest(tf.test.TestCase):
  """This class defines integration test cases for tensorflow_model_server."""

  def __TestSrcDirPath(self, relative_path):
    return os.path.join(os.environ['TEST_SRCDIR'],
                        'tf_serving/tensorflow_serving', relative_path)

  def setUp(self):
    """Sets up integration test parameters."""
    self.binary_dir = self.__TestSrcDirPath('model_servers')
    self.testdata_dir = self.__TestSrcDirPath('servables/tensorflow/testdata')
    self.server_proc = None

  def TerminateProcs(self):
    """Terminate all processes."""
    print 'Terminating all processes...'
    if self.server_proc is not None:
      self.server_proc.terminate()

  def RunServer(self, port, model_name, model_path):
    """Run tensorflow_model_server using test config."""
    print 'Starting test server...'
    command = os.path.join(self.binary_dir, 'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --model_name=' + model_name
    command += ' --model_base_path=' + model_path
    command += ' --alsologtostderr'
    print command
    self.server_proc = subprocess.Popen(shlex.split(command))
    print 'Server started'
    return 'localhost:' + str(port)

  def VerifyPredictRequest(self,
                           model_server_address,
                           specify_output=True):
    """Send PredictionService.Predict request and verify output."""
    print 'Sending Predict request...'
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'default'
    request.inputs['x'].dtype = types_pb2.DT_FLOAT
    request.inputs['x'].float_val.append(2.0)
    if specify_output:
      request.output_filter.append('y')
    # Send request
    host, port = model_server_address.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Predict(request, 5.0)  # 5 secs timeout
    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    self.assertEquals(1, len(result.outputs['y'].float_val))
    self.assertEquals(3.0, result.outputs['y'].float_val[0])

  def testPredict(self):
    """Test PredictionService.Predict implementation."""
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(
        PickUnusedPort(), 'default',
        os.path.join(self.testdata_dir, 'half_plus_two'))
    time.sleep(5)
    self.VerifyPredictRequest(model_server_address)
    self.VerifyPredictRequest(model_server_address, specify_output=False)

  def testBadModel(self):
    """Test PredictionService.Predict against a bad model export."""
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(
        PickUnusedPort(), 'default',
        os.path.join(self.testdata_dir, 'bad_half_plus_two'))
    time.sleep(5)
    with self.assertRaises(face.AbortionError) as error:
      self.VerifyPredictRequest(model_server_address)
    self.assertIs(StatusCode.FAILED_PRECONDITION,
                  error.exception.code)
    self.assertTrue(error.exception.details.startswith(
        'Expected exactly one signatures proto'))


if __name__ == '__main__':
  tf.test.main()
