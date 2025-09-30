#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# This script will be run by the building process to generate key-value
# information that represents the status of the workspace. The output should be
# in the format:
#
# KEY1 VALUE1
# KEY2 VALUE2
#
# If the script exits with non-zero code, it's considered as a failure
# and the output will be discarded.

# if we're inside a git tree
if [ -d .git ] || git rev-parse --git-dir > /dev/null 2>&1; then
  git_rev=$(git rev-parse --short HEAD)
  if [[ $? != 0 ]];
  then
      exit 1
  fi
  echo "BUILD_SCM_REVISION ${git_rev}"
else
  echo "BUILD_SCM_REVISION no_git"
fi;
