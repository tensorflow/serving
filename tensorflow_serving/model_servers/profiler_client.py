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
"""Simple client to send profiling request to ModelServer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.profiler import profiler_client


def main(argv):
  server = argv[1] if len(argv) > 1 else 'localhost:8500'
  logdir = argv[2] if len(argv) > 2 else '/tmp'
  duration_ms = argv[3] if len(argv) > 3 else 2000
  profiler_client.trace(server, logdir, duration_ms)


if __name__ == '__main__':
  tf.compat.v1.app.run()
