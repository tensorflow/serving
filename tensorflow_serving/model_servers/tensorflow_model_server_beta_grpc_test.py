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
"""Minimal sanity tests for tensorflow_model_server with beta gRPC APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow.python.saved_model import signature_constants
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import regression_pb2

FLAGS = flags.FLAGS

RPC_TIMEOUT = 5.0
CHANNEL_WAIT_TIMEOUT = 5.0
WAIT_FOR_SERVER_READY_INT_SECS = 60


def PickUnusedPort():
  s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
  s.bind(('', 0))
  port = s.getsockname()[1]
  s.close()
  return port


def WaitForServerReady(port):
  """Waits for a server on the localhost to become ready."""
  for _ in range(0, WAIT_FOR_SERVER_READY_INT_SECS):
    time.sleep(1)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'intentionally_missing_model'

    try:
      # Send empty request to missing model
      channel = implementations.insecure_channel('localhost', port)
      stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
      stub.Predict(request, RPC_TIMEOUT)
    except Exception as e:  # pylint: disable=broad-except
      # Missing model error will have details containing 'Servable'
      if 'Servable' in e.details():
        print('Server is ready')
        break


class TensorflowModelServerTest(tf.test.TestCase):
  """This class defines integration test cases for tensorflow_model_server."""

  @staticmethod
  def __TestSrcDirPath(relative_path=''):
    return os.path.join(os.environ['TEST_SRCDIR'],
                        'tf_serving/tensorflow_serving', relative_path)

  @staticmethod
  def GetArgsKey(*args, **kwargs):
    return args + tuple(sorted(kwargs.items()))

  # Maps string key -> 2-tuple of 'host:port' string.
  model_servers_dict = {}

  @staticmethod
  def RunServer(model_name,
                model_path,
                model_config_file=None,
                batching_parameters_file=None,
                grpc_channel_arguments='',
                wait_for_server_ready=True,
                pipe=None):
    """Run tensorflow_model_server using test config.

    A unique instance of server is started for each set of arguments.
    If called with same arguments, handle to an existing server is
    returned.

    Args:
      model_name: Name of model.
      model_path: Path to model.
      model_config_file: Path to model config file.
      batching_parameters_file: Path to batching parameters.
      grpc_channel_arguments: Custom gRPC args for server.
      wait_for_server_ready: Wait for gRPC port to be ready.
      pipe: subpipe.PIPE object to read stderr from server.

    Returns:
      3-tuple (<Popen object>, <grpc host:port>, <rest host:port>).

    Raises:
      ValueError: when both model_path and config_file is empty.
    """
    args_key = TensorflowModelServerTest.GetArgsKey(**locals())
    if args_key in TensorflowModelServerTest.model_servers_dict:
      return TensorflowModelServerTest.model_servers_dict[args_key]
    port = PickUnusedPort()
    print(('Starting test server on port: {} for model_name: '
           '{}/model_config_file: {}'.format(port, model_name,
                                             model_config_file)))
    command = os.path.join(
        TensorflowModelServerTest.__TestSrcDirPath('model_servers'),
        'tensorflow_model_server')
    command += ' --port=' + str(port)

    if model_config_file:
      command += ' --model_config_file=' + model_config_file
    elif model_path:
      command += ' --model_name=' + model_name
      command += ' --model_base_path=' + model_path
    else:
      raise ValueError('Both model_config_file and model_path cannot be empty!')

    if batching_parameters_file:
      command += ' --enable_batching'
      command += ' --batching_parameters_file=' + batching_parameters_file
    if grpc_channel_arguments:
      command += ' --grpc_channel_arguments=' + grpc_channel_arguments
    print(command)
    proc = subprocess.Popen(shlex.split(command), stderr=pipe)
    atexit.register(proc.kill)
    print('Server started')
    if wait_for_server_ready:
      WaitForServerReady(port)
    hostports = (proc, 'localhost:' + str(port), None)
    TensorflowModelServerTest.model_servers_dict[args_key] = hostports
    return hostports

  def __BuildModelConfigFile(self):
    """Write a config file to disk for use in tests.

    Substitutes placeholder for test directory with test directory path
    in the configuration template file and writes it out to another file
    used by the test.
    """
    with open(self._GetGoodModelConfigTemplate(), 'r') as template_file:
      config = template_file.read().replace('${TEST_HALF_PLUS_TWO_DIR}',
                                            self._GetSavedModelBundlePath())
      config = config.replace('${TEST_HALF_PLUS_THREE_DIR}',
                              self._GetSavedModelHalfPlusThreePath())
    with open(self._GetGoodModelConfigFile(), 'w') as config_file:
      config_file.write(config)

  def setUp(self):
    """Sets up integration test parameters."""
    self.testdata_dir = TensorflowModelServerTest.__TestSrcDirPath(
        'servables/tensorflow/testdata')
    self.temp_dir = tf.test.get_temp_dir()
    self.server_proc = None
    self.__BuildModelConfigFile()

  def tearDown(self):
    """Deletes created configuration file."""
    os.remove(self._GetGoodModelConfigFile())

  def _MakeStub(self, hostport):
    """Returns a gRPC stub using beta gRPC API."""
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    return prediction_service_pb2.beta_create_PredictionService_stub(channel)

  def VerifyPredictRequest(
      self,
      model_server_address,
      expected_output,
      expected_version,
      model_name='default',
      specify_output=True,
      signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Send PredictionService.Predict request and verify output."""
    print('Sending Predict request...')
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs['x'].dtype = types_pb2.DT_FLOAT
    request.inputs['x'].float_val.append(2.0)
    dim = request.inputs['x'].tensor_shape.dim.add()
    dim.size = 1

    if specify_output:
      request.output_filter.append('y')
    # Send request
    result = self._MakeStub(model_server_address).Predict(request, RPC_TIMEOUT)

    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    self.assertEqual(1, len(result.outputs['y'].float_val))
    self.assertEqual(expected_output, result.outputs['y'].float_val[0])
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          signature_name, expected_version)

  def _GetSavedModelBundlePath(self):
    """Returns a path to a model in SavedModel format."""
    return os.path.join(os.environ['TEST_SRCDIR'], 'tf_serving/external/org_tensorflow/tensorflow/',
                        'cc/saved_model/testdata/half_plus_two')

  def _GetModelVersion(self, model_path):
    """Returns version of SavedModel/SessionBundle in given path.

    This method assumes there is exactly one directory with an 'int' valued
    directory name under `model_path`.

    Args:
      model_path: A string representing path to the SavedModel/SessionBundle.

    Returns:
      version of SavedModel/SessionBundle in given path.
    """
    return int(os.listdir(model_path)[0])

  def _GetSavedModelHalfPlusThreePath(self):
    """Returns a path to a half_plus_three model in SavedModel format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_three')

  def _GetSessionBundlePath(self):
    """Returns a path to a model in SessionBundle format."""
    return os.path.join(self.testdata_dir, 'half_plus_two')

  def _GetGoodModelConfigTemplate(self):
    """Returns a path to a working configuration file template."""
    return os.path.join(self.testdata_dir, 'good_model_config.txt')

  def _GetGoodModelConfigFile(self):
    """Returns a path to a working configuration file."""
    return os.path.join(self.temp_dir, 'good_model_config.conf')

  def _VerifyModelSpec(self, actual_model_spec, exp_model_name,
                       exp_signature_name, exp_version):
    """Verifies model_spec matches expected model name, signature, version.

    Args:
      actual_model_spec: An instance of ModelSpec proto.
      exp_model_name: A string that represents expected model name.
      exp_signature_name: A string that represents expected signature.
      exp_version: An integer that represents expected version.

    Returns:
      None.
    """
    self.assertEqual(actual_model_spec.name, exp_model_name)
    self.assertEqual(actual_model_spec.signature_name, exp_signature_name)
    self.assertEqual(actual_model_spec.version.value, exp_version)

  def testClassify(self):
    """Test PredictionService.Classify implementation."""
    model_path = self._GetSavedModelBundlePath()
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path)[1]

    print('Sending Classify request...')
    # Prepare request
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'classify_x_to_y'

    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend([2.0])

    # Send request
    result = self._MakeStub(model_server_address).Classify(request, RPC_TIMEOUT)

    # Verify response
    self.assertEqual(1, len(result.result.classifications))
    self.assertEqual(1, len(result.result.classifications[0].classes))
    expected_output = 3.0
    self.assertEqual(expected_output,
                     result.result.classifications[0].classes[0].score)
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          request.model_spec.signature_name,
                          self._GetModelVersion(model_path))

  def testRegress(self):
    """Test PredictionService.Regress implementation."""
    model_path = self._GetSavedModelBundlePath()
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path)[1]

    print('Sending Regress request...')
    # Prepare request
    request = regression_pb2.RegressionRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'regress_x_to_y'

    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend([2.0])

    # Send request
    result = self._MakeStub(model_server_address).Regress(request, RPC_TIMEOUT)

    # Verify response
    self.assertEqual(1, len(result.result.regressions))
    expected_output = 3.0
    self.assertEqual(expected_output, result.result.regressions[0].value)
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          request.model_spec.signature_name,
                          self._GetModelVersion(model_path))

  def testMultiInference(self):
    """Test PredictionService.MultiInference implementation."""
    model_path = self._GetSavedModelBundlePath()
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path)[1]

    print('Sending MultiInference request...')
    # Prepare request
    request = inference_pb2.MultiInferenceRequest()
    request.tasks.add().model_spec.name = 'default'
    request.tasks[0].model_spec.signature_name = 'regress_x_to_y'
    request.tasks[0].method_name = 'tensorflow/serving/regress'
    request.tasks.add().model_spec.name = 'default'
    request.tasks[1].model_spec.signature_name = 'classify_x_to_y'
    request.tasks[1].method_name = 'tensorflow/serving/classify'

    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend([2.0])

    # Send request
    result = self._MakeStub(model_server_address).MultiInference(
        request, RPC_TIMEOUT)

    # Verify response
    self.assertEqual(2, len(result.results))
    expected_output = 3.0
    self.assertEqual(expected_output,
                     result.results[0].regression_result.regressions[0].value)
    self.assertEqual(
        expected_output, result.results[1].classification_result
        .classifications[0].classes[0].score)
    for i in range(2):
      self._VerifyModelSpec(result.results[i].model_spec,
                            request.tasks[i].model_spec.name,
                            request.tasks[i].model_spec.signature_name,
                            self._GetModelVersion(model_path))

  def testPredict(self):
    """Test PredictionService.Predict implementation with SavedModel."""
    model_path = self._GetSavedModelBundlePath()
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path)[1]
    expected_version = self._GetModelVersion(model_path)
    signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        expected_version=expected_version,
        signature_name=signature_name)
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        specify_output=False,
        expected_version=expected_version,
        signature_name=signature_name)


if __name__ == '__main__':
  tf.test.main()
