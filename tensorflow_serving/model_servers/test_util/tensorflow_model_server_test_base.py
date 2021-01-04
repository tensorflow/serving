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
import time

# This is a placeholder for a Google-internal import.

import grpc
from six.moves import range
from six.moves import urllib
import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow.python.saved_model import signature_constants
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

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
  physical_devices = tf.config.experimental.list_physical_devices('CPU')
  if not physical_devices:
    raise RuntimeError('No CPUs found')
  configs = tf.config.experimental.get_virtual_device_configuration(
      physical_devices[0])
  if configs is None:
    virtual_devices = [tf.config.experimental.VirtualDeviceConfiguration()
                       for _ in range(num_virtual_cpus)]
    tf.config.experimental.set_virtual_device_configuration(
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


class TensorflowModelServerTestBase(tf.test.TestCase):
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
    args_key = TensorflowModelServerTestBase.GetArgsKey(**locals())
    if args_key in TensorflowModelServerTestBase.model_servers_dict:
      return TensorflowModelServerTestBase.model_servers_dict[args_key]
    port = PickUnusedPort()
    rest_api_port = PickUnusedPort()
    print(('Starting test server on port: {} for model_name: '
           '{}/model_config_file: {}'.format(port, model_name,
                                             model_config_file)))
    command = os.path.join(
        TensorflowModelServerTestBase.__TestSrcDirPath('model_servers'),
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
      command += ' --prefer_tflite_model=true'

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
    TensorflowModelServerTestBase.model_servers_dict[args_key] = hostports
    return hostports

  def VerifyPredictRequest(
      self,
      model_server_address,
      expected_output,
      expected_version,
      model_name='default',
      specify_output=True,
      batch_input=False,
      signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      rpc_timeout=RPC_TIMEOUT):
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
    result = stub.Predict(request, rpc_timeout)  # 5 secs timeout
    # Verify response
    self.assertTrue('y' in result.outputs)
    self.assertEqual(types_pb2.DT_FLOAT, result.outputs['y'].dtype)
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

  def _GetSavedModelHalfPlusTwoTf2(self):
    """Returns a path to a TF2 half_plus_two model in SavedModel format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_two_tf2_cpu')

  def _GetSavedModelHalfPlusThreePath(self):
    """Returns a path to a half_plus_three model in SavedModel format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_three')

  def _GetTfLiteModelPath(self):
    """Returns a path to a model in TF Lite format."""
    return os.path.join(self.testdata_dir, 'saved_model_half_plus_two_tflite')

  def _GetTfLiteModelWithSigDefPath(self):
    """Returns a path to a model in TF Lite format."""
    return os.path.join(self.testdata_dir,
                        'saved_model_half_plus_two_tflite_with_sigdef')

  def _GetSessionBundlePath(self):
    """Returns a path to a model in SessionBundle format."""
    return os.path.join(self.session_bundle_testdata_dir, 'half_plus_two')

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
    model_server_address = TensorflowModelServerTestBase.RunServer(
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
