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

"""Tests for tensorflow_model_server."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import json
import os
import shlex
import socket
import subprocess
import sys
import time

# This is a placeholder for a Google-internal import.

import grpc
from six.moves import range
from six.moves import urllib
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import config as device_config
from tensorflow.python.platform import flags
from tensorflow.python.saved_model import signature_constants
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import regression_pb2

FLAGS = flags.FLAGS

RPC_TIMEOUT = 5.0
HTTP_REST_TIMEOUT_MS = 5000
CHANNEL_WAIT_TIMEOUT = 5.0
WAIT_FOR_SERVER_READY_INT_SECS = 60
GRPC_SOCKET_PATH = '/tmp/tf-serving.sock'


def SetVirtualCpus(num_virtual_cpus):
  """Create virtual CPU devices if they haven't yet been created."""
  if num_virtual_cpus < 1:
    raise ValueError('`num_virtual_cpus` must be at least 1 not %r' %
                     (num_virtual_cpus,))
  physical_devices = device_config.list_physical_devices('CPU')
  if not physical_devices:
    raise RuntimeError('No CPUs found')
  configs = device_config.get_virtual_device_configuration(physical_devices[0])
  if configs is None:
    virtual_devices = [context.VirtualDeviceConfiguration()
                       for _ in range(num_virtual_cpus)]
    device_config.set_virtual_device_configuration(
        physical_devices[0], virtual_devices)
  else:
    if len(configs) < num_virtual_cpus:
      raise RuntimeError('Already configured with %d < %d virtual CPUs' %
                         (len(configs), num_virtual_cpus))


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
      channel = grpc.insecure_channel('localhost:{}'.format(port))
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      stub.Predict(request, RPC_TIMEOUT)
    except grpc.RpcError as error:
      # Missing model error will have details containing 'Servable'
      if 'Servable' in error.details():
        print('Server is ready')
        break


def CallREST(url, req, max_attempts=60):
  """Returns HTTP response body from a REST API call."""
  for attempt in range(max_attempts):
    try:
      print('Attempt {}: Sending request to {} with data:\n{}'.format(
          attempt, url, req))
      json_data = json.dumps(req).encode('utf-8') if req is not None else None
      resp = urllib.request.urlopen(urllib.request.Request(url, data=json_data))
      resp_data = resp.read()
      print('Received response:\n{}'.format(resp_data))
      resp.close()
      return resp_data
    except Exception as e:  # pylint: disable=broad-except
      print('Failed attempt {}. Error: {}'.format(attempt, e))
      if attempt == max_attempts - 1:
        raise
      print('Retrying...')
      time.sleep(1)


def SortedObject(obj):
  """Returns sorted object (with nested list/dictionaries)."""
  if isinstance(obj, dict):
    return sorted((k, SortedObject(v)) for k, v in obj.items())
  if isinstance(obj, list):
    return sorted(SortedObject(x) for x in obj)
  if isinstance(obj, tuple):
    return list(sorted(SortedObject(x) for x in obj))
  else:
    return obj


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
                model_type='tf',
                model_config_file=None,
                monitoring_config_file=None,
                batching_parameters_file=None,
                grpc_channel_arguments='',
                wait_for_server_ready=True,
                pipe=None,
                model_config_file_poll_period=None):
    """Run tensorflow_model_server using test config.

    A unique instance of server is started for each set of arguments.
    If called with same arguments, handle to an existing server is
    returned.

    Args:
      model_name: Name of model.
      model_path: Path to model.
      model_type: Type of model TensorFlow ('tf') or TF Lite ('tflite').
      model_config_file: Path to model config file.
      monitoring_config_file: Path to the monitoring config file.
      batching_parameters_file: Path to batching parameters.
      grpc_channel_arguments: Custom gRPC args for server.
      wait_for_server_ready: Wait for gRPC port to be ready.
      pipe: subpipe.PIPE object to read stderr from server.
      model_config_file_poll_period: Period for polling the
      filesystem to discover new model configs.

    Returns:
      3-tuple (<Popen object>, <grpc host:port>, <rest host:port>).

    Raises:
      ValueError: when both model_path and config_file is empty.
    """
    args_key = TensorflowModelServerTest.GetArgsKey(**locals())
    if args_key in TensorflowModelServerTest.model_servers_dict:
      return TensorflowModelServerTest.model_servers_dict[args_key]
    port = PickUnusedPort()
    rest_api_port = PickUnusedPort()
    print(('Starting test server on port: {} for model_name: '
           '{}/model_config_file: {}'.format(port, model_name,
                                             model_config_file)))
    command = os.path.join(
        TensorflowModelServerTest.__TestSrcDirPath('model_servers'),
        'tensorflow_model_server')
    command += ' --port=' + str(port)
    command += ' --rest_api_port=' + str(rest_api_port)
    command += ' --rest_api_timeout_in_ms=' + str(HTTP_REST_TIMEOUT_MS)
    command += ' --grpc_socket_path=' + GRPC_SOCKET_PATH

    if model_config_file:
      command += ' --model_config_file=' + model_config_file
    elif model_path:
      command += ' --model_name=' + model_name
      command += ' --model_base_path=' + model_path
    else:
      raise ValueError('Both model_config_file and model_path cannot be empty!')

    if model_type == 'tflite':
      command += ' --use_tflite_model=true'

    if monitoring_config_file:
      command += ' --monitoring_config_file=' + monitoring_config_file

    if model_config_file_poll_period is not None:
      command += ' --model_config_file_poll_wait_seconds=' + str(
          model_config_file_poll_period)

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
    hostports = (
        proc,
        'localhost:' + str(port),
        'localhost:' + str(rest_api_port),
    )
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

  def VerifyPredictRequest(
      self,
      model_server_address,
      expected_output,
      expected_version,
      model_name='default',
      specify_output=True,
      batch_input=False,
      signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Send PredictionService.Predict request and verify output."""
    print('Sending Predict request...')
    # Prepare request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['x'].dtype = types_pb2.DT_FLOAT
    request.inputs['x'].float_val.append(2.0)
    dim = request.inputs['x'].tensor_shape.dim.add()
    dim.size = 1
    if batch_input:
      request.inputs['x'].tensor_shape.dim.add().size = 1

    if specify_output:
      request.output_filter.append('y')
    # Send request
    channel = grpc.insecure_channel(model_server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Predict(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertIs(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
    self.assertEqual(1, len(result.outputs['y'].float_val))
    self.assertEqual(expected_output, result.outputs['y'].float_val[0])
    self._VerifyModelSpec(result.model_spec, request.model_spec.name,
                          signature_name, expected_version)

  def _GetSavedModelBundlePath(self):
    """Returns a path to a model in SavedModel format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_two_cpu')

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

  def _GetTfLiteModelPath(self):
    """Returns a path to a model in TF Lite format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_two_tflite')

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

  def _GetModelMetadataFile(self):
    """Returns a path to a sample model metadata file."""
    return os.path.join(self.testdata_dir, 'half_plus_two_model_metadata.json')

  def _GetMonitoringConfigFile(self):
    """Returns a path to a monitoring configuration file."""
    return os.path.join(self.testdata_dir, 'monitoring_config.txt')

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
    self.assertEqual(actual_model_spec.name, exp_model_name)
    self.assertEqual(actual_model_spec.signature_name, exp_signature_name)
    self.assertEqual(actual_model_spec.version.value, exp_version)

  def testGetModelStatus(self):
    """Test ModelService.GetModelStatus implementation."""
    model_path = self._GetSavedModelBundlePath()
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path)[1]

    print('Sending GetModelStatus request...')
    # Send request
    request = get_model_status_pb2.GetModelStatusRequest()
    request.model_spec.name = 'default'
    channel = grpc.insecure_channel(model_server_address)
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    result = stub.GetModelStatus(request, RPC_TIMEOUT)  # 5 secs timeout
    # Verify response
    self.assertEqual(1, len(result.model_version_status))
    self.assertEqual(123, result.model_version_status[0].version)
    # OK error code (0) indicates no error occurred
    self.assertEqual(0, result.model_version_status[0].status.error_code)

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
    channel = grpc.insecure_channel(model_server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Classify(request, RPC_TIMEOUT)  # 5 secs timeout
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
    channel = grpc.insecure_channel(model_server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Regress(request, RPC_TIMEOUT)  # 5 secs timeout
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
    channel = grpc.insecure_channel(model_server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.MultiInference(request, RPC_TIMEOUT)  # 5 secs timeout

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

  def _TestPredict(
      self,
      model_path,
      batching_parameters_file=None,
      signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
    """Helper method to test prediction.

    Args:
      model_path:      Path to the model on disk.
      batching_parameters_file: Batching parameters file to use (if None
                                batching is not enabled).
      signature_name: Signature name to expect in the PredictResponse.
    """
    model_server_address = TensorflowModelServerTest.RunServer(
        'default',
        model_path,
        batching_parameters_file=batching_parameters_file)[1]
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
    # Both SessionBundle and SavedModel use the same bad model path, but in the
    # case of SavedModel, the export will get up-converted to a SavedModel.
    # As the bad model will prevent the server from becoming ready, we set the
    # wait_for_server_ready param to False to avoid blocking/timing out.
    model_path = os.path.join(self.testdata_dir, 'bad_half_plus_two'),
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', model_path, wait_for_server_ready=False)[1]
    with self.assertRaises(grpc.RpcError) as ectxt:
      self.VerifyPredictRequest(
          model_server_address, expected_output=3.0,
          expected_version=self._GetModelVersion(model_path),
          signature_name='')
    self.assertIs(grpc.StatusCode.FAILED_PRECONDITION,
                  ectxt.exception.code())

  def _TestBadModelUpconvertedSavedModel(self):
    """Test Predict against a bad upconverted SavedModel model export."""
    self._TestBadModel()

  def testGoodModelConfig(self):
    """Test server configuration from file works with valid configuration."""
    model_server_address = TensorflowModelServerTest.RunServer(
        None, None, model_config_file=self._GetGoodModelConfigFile())[1]

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
    proc = TensorflowModelServerTest.RunServer(
        None,
        None,
        model_config_file=self._GetBadModelConfigFile(),
        pipe=subprocess.PIPE,
        wait_for_server_ready=False)[0]

    error_message = ('Error parsing text-format '
                     'tensorflow.serving.ModelServerConfig')
    error_message = error_message.encode('utf-8')
    self.assertNotEqual(proc.stderr, None)
    self.assertGreater(proc.stderr.read().find(error_message), -1)

  def testModelConfigReload(self):
    """Test model server polls filesystem for model configuration."""

    base_config_proto = """
    model_config_list: {{
      config: {{
        name: "{name}",
        base_path: "{model_path}",
        model_platform: "tensorflow"
      }}
    }}
    """

    config_path = os.path.join(FLAGS.test_tmpdir, 'model_config.txt')

    # Write a config file serving half_plus_two model
    with open(config_path, 'w') as f:
      f.write(
          base_config_proto.format(
              name='half_plus_two', model_path=self._GetSavedModelBundlePath()))

    poll_period = 1
    model_server_address = TensorflowModelServerTest.RunServer(
        None,
        None,
        model_config_file=config_path,
        model_config_file_poll_period=poll_period)[1]

    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_two',
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))

    # Rewrite the config file with half_plus_three model
    with open(config_path, 'w') as f:
      f.write(
          base_config_proto.format(
              name='half_plus_three',
              model_path=self._GetSavedModelHalfPlusThreePath()))

    # Give modelserver time to poll and load the new config
    time.sleep(poll_period + 1)

    # Verify new model config was realized in model server
    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_three',
        expected_output=4.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

  def testModelConfigReloadWithZeroPollPeriod(self):
    """Test model server does not poll filesystem for model config."""

    base_config_proto = """
    model_config_list: {{
      config: {{
        name: "{name}",
        base_path: "{model_path}",
        model_platform: "tensorflow"
      }}
    }}
    """

    config_path = os.path.join(FLAGS.test_tmpdir, 'model_config.txt')

    # Write a config file serving half_plus_two model
    with open(config_path, 'w') as f:
      f.write(
          base_config_proto.format(
              name='half_plus_two', model_path=self._GetSavedModelBundlePath()))

    poll_period = 0
    model_server_address = TensorflowModelServerTest.RunServer(
        None,
        None,
        model_config_file=config_path,
        model_config_file_poll_period=poll_period)[1]

    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_two',
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(self._GetSavedModelBundlePath()))

    # Rewrite the config file with half_plus_three model
    with open(config_path, 'w') as f:
      f.write(
          base_config_proto.format(
              name='half_plus_three',
              model_path=self._GetSavedModelHalfPlusThreePath()))

    # Give modelserver enough time to poll and load the new config should it
    # have such a desire
    time.sleep(poll_period + 1)

    # Verify model server is still serving the old model config
    self.VerifyPredictRequest(
        model_server_address,
        model_name='half_plus_two',
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

  def testGoodGrpcChannelArgs(self):
    """Test server starts with grpc_channel_arguments specified."""
    model_server_address = TensorflowModelServerTest.RunServer(
        'default',
        self._GetSavedModelBundlePath(),
        grpc_channel_arguments=
        'grpc.max_connection_age_ms=2000,grpc.lb_policy_name=grpclb')[1]
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

  def testClassifyREST(self):
    """Test Classify implementation over REST API."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default:classify'.format(host, port)
    json_req = {'signature_name': 'classify_x_to_y', 'examples': [{'x': 2.0}]}

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, json_req)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify response
    self.assertEqual(json.loads(resp_data), {'results': [[['', 3.0]]]})

  def testRegressREST(self):
    """Test Regress implementation over REST API."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default:regress'.format(host, port)
    json_req = {'signature_name': 'regress_x_to_y', 'examples': [{'x': 2.0}]}

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, json_req)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify response
    self.assertEqual(json.loads(resp_data), {'results': [3.0]})

  def testPredictREST(self):
    """Test Predict implementation over REST API."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default:predict'.format(host, port)
    json_req = {'instances': [2.0, 3.0, 4.0]}

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, json_req)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify response
    self.assertEqual(json.loads(resp_data), {'predictions': [3.0, 3.5, 4.0]})

  def testPredictColumnarREST(self):
    """Test Predict implementation over REST API with columnar inputs."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default:predict'.format(host, port)
    json_req = {'inputs': [2.0, 3.0, 4.0]}

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, json_req)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify response
    self.assertEqual(json.loads(resp_data), {'outputs': [3.0, 3.5, 4.0]})

  def testGetStatusREST(self):
    """Test ModelStatus implementation over REST API with columnar inputs."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default'.format(host, port)

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, None)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify response
    self.assertEqual(
        json.loads(resp_data), {
            'model_version_status': [{
                'version': '123',
                'state': 'AVAILABLE',
                'status': {
                    'error_code': 'OK',
                    'error_message': ''
                }
            }]
        })

  def testGetModelMetadataREST(self):
    """Test ModelStatus implementation over REST API with columnar inputs."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer('default',
                                                     model_path)[2].split(':')

    # Prepare request
    url = 'http://{}:{}/v1/models/default/metadata'.format(host, port)

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, None)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    try:
      model_metadata_file = self._GetModelMetadataFile()
      with open(model_metadata_file) as f:
        expected_metadata = json.load(f)
        # Verify response
        # Note, we sort the JSON object before comparing. Formally JSON lists
        # (aka arrays) are considered ordered and general comparison should NOT
        # sort. In this case, the "metadata" does not have any ordering making
        # the sort OK (and the test robust).
        self.assertEqual(
            SortedObject(json.loads(resp_data)),
            SortedObject(expected_metadata))
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

  def testPrometheusEndpoint(self):
    """Test ModelStatus implementation over REST API with columnar inputs."""
    model_path = self._GetSavedModelBundlePath()
    host, port = TensorflowModelServerTest.RunServer(
        'default',
        model_path,
        monitoring_config_file=self._GetMonitoringConfigFile())[2].split(':')

    # Prepare request
    url = 'http://{}:{}/monitoring/prometheus/metrics'.format(host, port)

    # Send request
    resp_data = None
    try:
      resp_data = CallREST(url, None)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Request failed with error: {}'.format(e))

    # Verify that there should be some metric type information.
    self.assertIn('# TYPE',
                  resp_data.decode('utf-8') if resp_data is not None else None)

  def testPredictUDS(self):
    """Test saved model prediction over a Unix domain socket."""
    _ = TensorflowModelServerTest.RunServer('default',
                                            self._GetSavedModelBundlePath())
    model_server_address = 'unix:%s' % GRPC_SOCKET_PATH
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(
            self._GetSavedModelHalfPlusThreePath()))

  def testPredictOnTfLite(self):
    """Test saved model prediction on a TF Lite mode."""
    model_server_address = TensorflowModelServerTest.RunServer(
        'default', self._GetTfLiteModelPath(), model_type='tflite')[1]
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=3.0,
        specify_output=False,
        expected_version=self._GetModelVersion(self._GetTfLiteModelPath()))

  def test_tf_saved_model_save(self):
    base_path = os.path.join(self.get_temp_dir(), 'tf_saved_model_save')
    export_path = os.path.join(base_path, '00000123')
    root = tf.train.Checkpoint()
    root.v1 = tf.Variable(3.)
    root.v2 = tf.Variable(2.)
    root.f = tf.function(
        lambda x: {'y': root.v1 * root.v2 * x})
    to_save = root.f.get_concrete_function(tf.TensorSpec(None, tf.float32))
    tf.saved_model.experimental.save(root, export_path, to_save)
    _, model_server_address, _ = TensorflowModelServerTest.RunServer(
        'default', base_path)
    expected_version = self._GetModelVersion(base_path)
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=12.0,
        specify_output=False,
        expected_version=expected_version)

  def test_tf_saved_model_save_multiple_signatures(self):
    base_path = os.path.join(self.get_temp_dir(), 'tf_saved_model_save')
    export_path = os.path.join(base_path, '00000123')
    root = tf.train.Checkpoint()
    root.f = tf.function(lambda x: {'y': 1.},
                         input_signature=[tf.TensorSpec(None, tf.float32)])
    root.g = tf.function(lambda x: {'y': 2.},
                         input_signature=[tf.TensorSpec(None, tf.float32)])
    tf.saved_model.experimental.save(
        root, export_path,
        signatures={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.f,
            'custom_signature_key': root.g})
    _, model_server_address, _ = TensorflowModelServerTest.RunServer(
        'default', base_path)
    expected_version = self._GetModelVersion(base_path)
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=2.0,
        expected_version=expected_version,
        signature_name='custom_signature_key')
    self.VerifyPredictRequest(
        model_server_address,
        expected_output=1.0,
        expected_version=expected_version)

  def test_sequential_keras_saved_model_save(self):
    """Test loading a simple SavedModel created with Keras Sequential API."""
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(dtype='float32', shape=(1,), name='x'))
    model.add(tf.keras.layers.Lambda(lambda x: x, name='y'))

    base_path = os.path.join(self.get_temp_dir(),
                             'keras_sequential_saved_model_save')
    export_path = os.path.join(base_path, '00000123')

    tf.saved_model.save(model, export_path)

    _, model_server_address, _ = TensorflowModelServerTest.RunServer(
        'default', base_path)

    expected_version = self._GetModelVersion(base_path)
    self.VerifyPredictRequest(
        model_server_address,
        batch_input=True,
        specify_output=False,
        expected_output=2.0,
        expected_version=expected_version)

  def test_distrat_sequential_keras_saved_model_save(self):
    """Test loading a Keras SavedModel with tf.distribute."""
    # You need to call SetVirtualCpus in test setUp with the maximum value
    # needed in any test if you use this in multiple tests. For now this is the
    # only test using this functionality.
    SetVirtualCpus(2)
    strategy = tf.distribute.MirroredStrategy(devices=('/cpu:0', '/cpu:1'))
    with strategy.scope():
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Input(dtype='float32', shape=(1,), name='x'))
      model.add(tf.keras.layers.Dense(1, kernel_initializer='ones',
                                      bias_initializer='zeros'))
      model.add(tf.keras.layers.Lambda(lambda x: x, name='y'))

    base_path = os.path.join(self.get_temp_dir(),
                             'keras_sequential_saved_model_save')
    export_path = os.path.join(base_path, '00000123')

    tf.saved_model.save(model, export_path)

    _, model_server_address, _ = TensorflowModelServerTest.RunServer(
        'default', base_path)

    expected_version = self._GetModelVersion(base_path)
    self.VerifyPredictRequest(
        model_server_address,
        batch_input=True,
        specify_output=False,
        expected_output=2.0,
        expected_version=expected_version)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
