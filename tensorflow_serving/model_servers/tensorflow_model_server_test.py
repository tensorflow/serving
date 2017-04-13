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

from grpc.beta import implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.interfaces.face import face
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import inference_pb2

FLAGS = flags.FLAGS


def PickUnusedPort():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(('localhost', 0))
  _, port = s.getsockname()
  s.close()
  return port


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

  def RunServer(self, port, model_name, model_path, use_saved_model,
                batching_parameters_file=''):
    """Run tensorflow_model_server using test config."""
    print 'Starting test server...'
    command = os.path.join(self.binary_dir, 'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --model_name=' + model_name
    command += ' --model_base_path=' + model_path
    command += ' --use_saved_model=' + str(use_saved_model).lower()
    if batching_parameters_file:
      command += ' --enable_batching'
      command += ' --batching_parameters_file=' + batching_parameters_file
    print command
    self.server_proc = subprocess.Popen(shlex.split(command))
    print 'Server started'
    return 'localhost:' + str(port)

  def RunServerWithModelConfigFile(self,
                                   port,
                                   model_config_file,
                                   use_saved_model,
                                   pipe=None):
    """Run tensorflow_model_server using test config."""
    print 'Starting test server...'
    command = os.path.join(self.binary_dir, 'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --model_config_file=' + model_config_file
    command += ' --use_saved_model=' + str(use_saved_model).lower()

    print command
    self.server_proc = subprocess.Popen(shlex.split(command), stderr=pipe)
    print 'Server started'
    return 'localhost:' + str(port)

  def VerifyPredictRequest(self,
                           model_server_address,
                           expected_output,
                           model_name='default',
                           specify_output=True):
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
    result = stub.Predict(request, 5.0)  # 5 secs timeout
    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    self.assertEquals(1, len(result.outputs['y'].float_val))
    self.assertEquals(expected_output, result.outputs['y'].float_val[0])

  def _GetSavedModelBundlePath(self):
    """Returns a path to a model in SavedModel format."""
    return os.path.join(os.environ['TEST_SRCDIR'], 'tf_serving/external/org_tensorflow/tensorflow/',
                        'cc/saved_model/testdata/half_plus_two')

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

  def testClassify(self):
    """Test PredictionService.Classify implementation."""
    model_path = self._GetSavedModelBundlePath()
    use_saved_model = True

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path, use_saved_model)
    time.sleep(5)

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
    result = stub.Classify(request, 5.0)  # 5 secs timeout
    # Verify response
    self.assertEquals(1, len(result.result.classifications))
    self.assertEquals(1, len(result.result.classifications[0].classes))
    expected_output = 3.0
    self.assertEquals(expected_output,
                      result.result.classifications[0].classes[0].score)

  def testRegress(self):
    """Test PredictionService.Regress implementation."""
    model_path = self._GetSavedModelBundlePath()
    use_saved_model = True

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path, use_saved_model)
    time.sleep(5)

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
    result = stub.Regress(request, 5.0)  # 5 secs timeout
    # Verify response
    self.assertEquals(1, len(result.result.regressions))
    expected_output = 3.0
    self.assertEquals(expected_output,
                      result.result.regressions[0].value)

  def testMultiInference(self):
    """Test PredictionService.MultiInference implementation."""
    model_path = self._GetSavedModelBundlePath()
    use_saved_model = True
    enable_batching = False

    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path, use_saved_model,
                                          enable_batching)
    time.sleep(5)

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
    result = stub.MultiInference(request, 5.0)  # 5 secs timeout

    # Verify response
    self.assertEquals(2, len(result.results))
    expected_output = 3.0
    self.assertEquals(expected_output,
                      result.results[0].regression_result.regressions[0].value)
    self.assertEquals(expected_output, result.results[
        1].classification_result.classifications[0].classes[0].score)

  def _TestPredict(self, model_path, use_saved_model,
                   batching_parameters_file=''):
    """Helper method to test prediction.

    Args:
      model_path:      Path to the model on disk.
      use_saved_model: Whether the model server should use SavedModel.
      batching_parameters_file: Batching parameters file to use (if left empty,
                                batching is not enabled).
    """
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServer(PickUnusedPort(), 'default',
                                          model_path, use_saved_model,
                                          batching_parameters_file)
    time.sleep(5)
    self.VerifyPredictRequest(model_server_address, expected_output=3.0)
    self.VerifyPredictRequest(
        model_server_address, expected_output=3.0, specify_output=False)

  def testPredictSessionBundle(self):
    """Test PredictionService.Predict implementation with SessionBundle."""
    self._TestPredict(self._GetSessionBundlePath(), use_saved_model=False)

  def testPredictBatchingSessionBundle(self):
    """Test PredictionService.Predict implementation with SessionBundle."""
    self._TestPredict(self._GetSessionBundlePath(),
                      use_saved_model=False,
                      batching_parameters_file=
                      self._GetBatchingParametersFile())

  def testPredictSavedModel(self):
    """Test PredictionService.Predict implementation with SavedModel."""
    self._TestPredict(self._GetSavedModelBundlePath(), use_saved_model=True)

  def testPredictUpconvertedSavedModel(self):
    """Test PredictionService.Predict implementation.

    Using a SessionBundle converted to a SavedModel.
    """
    self._TestPredict(self._GetSessionBundlePath(), use_saved_model=True)

  def _TestBadModel(self, use_saved_model):
    """Helper method to test against a bad model export."""
    atexit.register(self.TerminateProcs)
    # Both SessionBundle and SavedModel use the same bad model path, but in the
    # case of SavedModel, the export will get up-converted to a SavedModel.
    model_server_address = self.RunServer(
        PickUnusedPort(), 'default',
        os.path.join(self.testdata_dir, 'bad_half_plus_two'), use_saved_model)
    time.sleep(5)
    with self.assertRaises(face.AbortionError) as error:
      self.VerifyPredictRequest(model_server_address, expected_output=3.0)
    self.assertIs(beta_interfaces.StatusCode.FAILED_PRECONDITION,
                  error.exception.code)

  def _TestBadModelUpconvertedSavedModel(self):
    """Test Predict against a bad upconverted SavedModel model export."""
    self._TestBadModel(use_saved_model=True)

  def _TestBadModelSessionBundle(self):
    """Test Predict against a bad SessionBundle model export."""
    self._TestBadModel(use_saved_model=False)

  def testGoodModelConfig(self):
    """Test server configuration from file works with valid configuration."""
    atexit.register(self.TerminateProcs)
    model_server_address = self.RunServerWithModelConfigFile(
        PickUnusedPort(), self._GetGoodModelConfigFile(),
        True)  # use_saved_model
    time.sleep(5)

    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_two', expected_output=3.0)
    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_two',
        expected_output=3.0,
        specify_output=False)

    self.VerifyPredictRequest(
        model_server_address, model_name='half_plus_three', expected_output=4.0)
    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_three',
        expected_output=4.0,
        specify_output=False)

  def testBadModelConfig(self):
    """Test server model configuration from file fails for invalid file."""
    atexit.register(self.TerminateProcs)
    self.RunServerWithModelConfigFile(
        PickUnusedPort(),
        self._GetBadModelConfigFile(),
        True,  # use_saved_model
        pipe=subprocess.PIPE)

    error_message = (
        'Invalid protobuf file: \'%s\'') % self._GetBadModelConfigFile()
    self.assertNotEqual(self.server_proc.stderr, None)
    self.assertGreater(self.server_proc.stderr.read().find(error_message), -1)

if __name__ == '__main__':
  tf.test.main()
