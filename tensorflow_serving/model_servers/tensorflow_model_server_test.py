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

import grpc
from grpc.beta import implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.interfaces.face import face
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow.python.saved_model import signature_constants

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
    except face.AbortionError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details:
        print 'Server is ready'
        break


class TensorflowModelServerTest(tf.test.TestCase):
  """This class defines integration test cases for tensorflow_model_server."""

  def __TestSrcDirPath(self, relative_path=''):
    return os.path.join(os.environ['TEST_SRCDIR'],
                        'tf_serving/tensorflow_serving', relative_path)

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
    self.binary_dir = self.__TestSrcDirPath('model_servers')
    self.testdata_dir = self.__TestSrcDirPath('servables/tensorflow/testdata')
    self.temp_dir = tf.test.get_temp_dir()
    self.server_proc = None
    self.__BuildModelConfigFile()

  def tearDown(self):
    """Deletes created configuration file."""
    os.remove(self._GetGoodModelConfigFile())

  def TerminateProcs(self):
    """Terminate all processes."""
    print 'Terminating all processes...'
    if self.server_proc is not None:
      self.server_proc.terminate()

  def RunServer(self,
                port,
                model_name,
                model_path,
                batching_parameters_file='',
                grpc_channel_arguments='',
                wait_for_server_ready=True):
    """Run tensorflow_model_server using test config."""
    print 'Starting test server...'
    command = os.path.join(self.binary_dir, 'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --model_name=' + model_name
    command += ' --model_base_path=' + model_path
    if batching_parameters_file:
      command += ' --enable_batching'
      command += ' --batching_parameters_file=' + batching_parameters_file
    if grpc_channel_arguments:
      command += ' --grpc_channel_arguments=' + grpc_channel_arguments
    print command
    self.server_proc = subprocess.Popen(shlex.split(command))
    print 'Server started'
    if wait_for_server_ready:
      WaitForServerReady(port)
    return 'localhost:' + str(port)

  def RunServerWithModelConfigFile(self,
                                   port,
                                   model_config_file,
                                   pipe=None,
                                   wait_for_server_ready=True):
    """Run tensorflow_model_server using test config."""
    print 'Starting test server...'
    command = os.path.join(self.binary_dir, 'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --model_config_file=' + model_config_file

    print command
    self.server_proc = subprocess.Popen(shlex.split(command), stderr=pipe)
    print 'Server started'
    if wait_for_server_ready:
      WaitForServerReady(port)
    return 'localhost:' + str(port)

  def VerifyPredictRequest(self,
                           model_server_address,
                           expected_output,
                           expected_version,
                           model_name='default',
                           specify_output=True,
                           signature_name=
                           signature_constants.
                           DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Send PredictionService.Predict request and verify output."""
    print 'Sending Predict request...'
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
    host, port = model_server_address.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Predict(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    self.assertEquals(1, len(result.outputs['y'].float_val))
    self.assertEquals(expected_output, result.outputs['y'].float_val[0])
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

  def _GetBadModelConfigFile(self):
    """Returns a path to a improperly formatted configuration file."""
    return os.path.join(self.testdata_dir, 'bad_model_config.txt')

  def _GetBatchingParametersFile(self):
    """Returns a path to a batching configuration file."""
    return os.path.join(self.testdata_dir, 'batching_config.txt')

  def _VerifyModelSpec(self,
                       actual_model_spec,
                       exp_model_name,
                       exp_signature_name,
                       exp_version):
    """Verifies model_spec matches expected model name, signature, version.

    Args:
      actual_model_spec: An instance of ModelSpec proto.
      exp_model_name: A string that represents expected model name.
      exp_signature_name: A string that represents expected signature.
      exp_version: An integer that represents expected version.

    Returns:
      None.
    """
    self.assertEquals(actual_model_spec.name, exp_model_name)
    self.assertEquals(actual_model_spec.signature_name, exp_signature_name)
    self.assertEquals(actual_model_spec.version.value, exp_version)

  def testGetModelStatus(self):
    """Test ModelService.GetModelStatus implementation."""
    model_path = self._GetSavedModelBundlePath()

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path)

    print 'Sending GetModelStatus request...'
    # Send request
    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = 'default'
    channel = grpc.insecure_channel(model_server_address)
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    result = stub.GetModelStatus(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertEquals(1, len(result.model_version_status))
    self.assertEquals(123, result.model_version_status[0].version)
    # OK error code (0) indicates no error occurred
    self.assertEquals(0, result.model_version_status[0].status.error_code)

  def testClassify(self):
    """Test PredictionService.Classify implementation."""
    model_path = self._GetSavedModelBundlePath()

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path)

    print 'Sending Classify request...'
    # Prepare request
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'classify_x_to_y'

    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend([2.0])

    # Send request
    host, port = model_server_address.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Classify(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertEquals(1, len(result.result.classifications))
    self.assertEquals(1, len(result.result.classifications[0].classes))
    expected_output = 3.0
    self.assertEquals(expected_output,
                      result.result.classifications[0].classes[0].score)
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          request.model_spec.signature_name,
                          self._GetModelVersion(model_path))

  def testRegress(self):
    """Test PredictionService.Regress implementation."""
    model_path = self._GetSavedModelBundlePath()

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path)

    print 'Sending Regress request...'
    # Prepare request
    request = regression_pb2.RegressionRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = 'regress_x_to_y'

    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend([2.0])

    # Send request
    host, port = model_server_address.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Regress(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertEquals(1, len(result.result.regressions))
    expected_output = 3.0
    self.assertEquals(expected_output, result.result.regressions[0].value)
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          request.model_spec.signature_name,
                          self._GetModelVersion(model_path))

  def testMultiInference(self):
    """Test PredictionService.MultiInference implementation."""
    model_path = self._GetSavedModelBundlePath()
    enable_batching = False

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path,
                                          enable_batching)

    print 'Sending MultiInference request...'
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
    host, port = model_server_address.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.MultiInference(request, RPC_TIMEOUT)  # 5 secs timeout

    # Verify response
    self.assertEquals(2, len(result.results))
    expected_output = 3.0
    self.assertEquals(expected_output,
                      result.results[0].regression_result.regressions[0].value)
    self.assertEquals(expected_output, result.results[
        1].classification_result.classifications[0].classes[0].score)
    for i in xrange(2):
      self._VerifyModelSpec(result.results[i].model_spec,
                            request.tasks[i].model_spec.name,
                            request.tasks[i].model_spec.signature_name,
                            self._GetModelVersion(model_path))

  def _TestPredict(self,
                   model_path,
                   batching_parameters_file='',
                   signature_name=
                   signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Helper method to test prediction.

    Args:
      model_path:      Path to the model on disk.
      batching_parameters_file: Batching parameters file to use (if left empty,
                                batching is not enabled).
      signature_name: Signature name to expect in the PredictResponse.
    """
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path, batching_parameters_file)
    expected_version = self._GetModelVersion(model_path)
    self.VerifyPredictRequest(model_server_address, expected_output=3.0,
                              expected_version=expected_version,
                              signature_name=signature_name)
    self.VerifyPredictRequest(
        model_server_address, expected_output=3.0, specify_output=False,
        expected_version=expected_version, signature_name=signature_name)

  def testPredictBatching(self):
    """Test PredictionService.Predict implementation with SessionBundle."""
    self._TestPredict(
        self._GetSessionBundlePath(),
        batching_parameters_file=self._GetBatchingParametersFile())

  def testPredictSavedModel(self):
    """Test PredictionService.Predict implementation with SavedModel."""
    self._TestPredict(self._GetSavedModelBundlePath())

  def testPredictUpconvertedSavedModel(self):
    """Test PredictionService.Predict implementation.

    Using a SessionBundle converted to a SavedModel.
    """
    self._TestPredict(self._GetSessionBundlePath())

  def _TestBadModel(self):
    """Helper method to test against a bad model export."""
    atexit.register(self.TerminateProcs)
    # Both SessionBundle and SavedModel use the same bad model path, but in the
    # case of SavedModel, the export will get up-converted to a SavedModel.
    # As the bad model will prevent the server from becoming ready, we set the
    # wait_for_server_ready param to False to avoid blocking/timing out.
    model_path = os.path.join(self.testdata_dir, 'bad_half_plus_two'),
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path,
                                          wait_for_server_ready=False)
    with self.assertRaises(face.AbortionError) as error:
      self.VerifyPredictRequest(
          model_server_address, expected_output=3.0,
          expected_version=self._GetModelVersion(model_path),
          signature_name='')
    self.assertIs(beta_interfaces.StatusCode.FAILED_PRECONDITION,
                  error.exception.code)

  def _TestBadModelUpconvertedSavedModel(self):
    """Test Predict against a bad upconverted SavedModel model export."""
    self._TestBadModel()

  def testGoodModelConfig(self):
    """Test server configuration from file works with valid configuration."""
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServerWithModelConfigFile(
        PickUnusedPort(), self._GetGoodModelConfigFile())

    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_two', expected_output=3.0,
        expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))
    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_two',
        expected_output=3.0, specify_output=False,
        expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))

    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_three', expected_output=4.0,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))
    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_three', expected_output=4.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

  def testBadModelConfig(self):
    """Test server model configuration from file fails for invalid file."""
    atexit.register(self.TerminateProcs)
    self.RunServerWithModelConfigFile(
        PickUnusedPort(),
        self._GetBadModelConfigFile(),
        pipe=subprocess.PIPE,
        wait_for_server_ready=False)

    error_message = (
        'Invalid protobuf file: \'%s\'') % self._GetBadModelConfigFile()
    self.assertNotEqual(self.server_proc.stderr, None)
    self.assertGreater(self.server_proc.stderr.read().find(error_message), -1)

  def testGoodGrpcChannelArgs(self):
    """Test server starts with grpc_channel_arguments specified."""
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(
        PickUnusedPort(),
        'default',
        self._GetSavedModelBundlePath(),
        grpc_channel_arguments=
        'grpc.max_connection_age_ms=2000,grpc.lb_policy_name=grpclb')
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

if __name__ == '__main__':
  tf.test.main()
