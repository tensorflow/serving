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

"""Common Python test utils.
"""

import os.path
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS


def TestSrcDirPath(relative_path):
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow_serving/
      e.g. "session_bundle/example".

  Returns:
    An absolute path to the linked in runfiles given the relative path.
  """
  # "tf_serving" is the name of the Bazel workspace, and newer versions of Bazel
  # will include it in the runfiles path.
  base_path = os.path.join(os.environ['TEST_SRCDIR'],
                           "tf_serving/tensorflow_serving")
  if gfile.Exists(base_path):
    # Supported in Bazel 0.2.2+.
    return os.path.join(base_path, relative_path)
  # Old versions of Bazel sometimes don't include the workspace name in the
  # runfiles path.
  return os.path.join(os.environ['TEST_SRCDIR'],
                      "tensorflow_serving", relative_path)
