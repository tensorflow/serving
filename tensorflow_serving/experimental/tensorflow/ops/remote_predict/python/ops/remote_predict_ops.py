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
import tensorflow.google.compat.v1 as tf

from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops import gen_remote_predict_op
# pylint: disable=wildcard-import
from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops.gen_remote_predict_op import *
# pylint: enable=wildcard-import

_remote_predict_op_module = tf.load_op_library(
    os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 '_remote_predict_op.so'))

# Alias
run = gen_remote_predict_op.tf_serving_remote_predict
