# Copyright 2020 Google Inc. All Rights Reserved.
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
"""Operations for RemotePredict."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
# This is a placeholder for a Google-internal import.
import tensorflow.compat.v1 as tf

from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops import gen_remote_predict_op
# pylint: disable=wildcard-import
from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops.gen_remote_predict_op import *
# pylint: enable=wildcard-import

_remote_predict_op_module = tf.load_op_library(
    os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 '_remote_predict_op.so'))


# Aliases
def run(input_tensor_alias,
        input_tensors,
        output_tensor_alias,
        target_address,
        model_name,
        model_version=-1,
        max_rpc_deadline_millis=3000,
        output_types=None,
        name=None,
        signature_name='serving_default'):
  """Runs a predict in remote process through rpc.

  Args:
    input_tensor_alias: input tensor alias for Predict
    input_tensors: input tensors for Predict
    output_tensor_alias: output tensor alias for Predict
    target_address: target_address where the rpc is sent to
    model_name: model_name that the Predict is running on
    model_version: the model version for the Predict call. If unset, the highest
      version available for serving will be targeted.
    max_rpc_deadline_millis: rpc deadline in millis
    output_types: output types for Predict
    name: name for the op in the graph
    signature_name: the signature def for remote graph inference

  Returns:
    output_tensors as a result of the Predict.

  Raises ValueError if model_name value is missing.
  """
  if model_name is None:
    raise ValueError('model_name must be specified.')
  return (gen_remote_predict_op.tf_serving_remote_predict(
      input_tensor_alias,
      input_tensors,
      output_tensor_alias,
      target_address=target_address,
      model_name=model_name,
      model_version=model_version,
      fail_op_on_rpc_error=True,
      max_rpc_deadline_millis=max_rpc_deadline_millis,
      signature_name=signature_name,
      output_types=output_types,
      name=name))[2]


def run_returning_status(input_tensor_alias,
                         input_tensors,
                         output_tensor_alias,
                         target_address,
                         model_name,
                         model_version=-1,
                         max_rpc_deadline_millis=3000,
                         output_types=None,
                         name=None,
                         signature_name='serving_default'):
  """Runs a predict in remote process through rpc.

  Args:
    input_tensor_alias: input tensor alias for Predict
    input_tensors: input tensors for Predict
    output_tensor_alias: output tensor alias for Predict
    target_address: target_address where the rpc is sent to
    model_name: model_name that the Predict is running on
    model_version: the model version for the Predict call. If unset, the highest
      version available for serving will be targeted.
    max_rpc_deadline_millis: rpc deadline in millis
    output_types: output types for Predict
    name: name for the op in the graph
    signature_name: the signature def for remote graph inference

  Returns:
    status_code, status_error_message and output_tensors.

  Raises ValueError if model_name value is missing.
  """
  if model_name is None:
    raise ValueError('model_name must be specified.')
  return (gen_remote_predict_op.tf_serving_remote_predict(
      input_tensor_alias,
      input_tensors,
      output_tensor_alias,
      target_address=target_address,
      model_name=model_name,
      model_version=model_version,
      fail_op_on_rpc_error=False,
      max_rpc_deadline_millis=max_rpc_deadline_millis,
      signature_name=signature_name,
      output_types=output_types,
      name=name))
