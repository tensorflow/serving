#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

function main() {
  if [[ $# -lt 1 ]] ; then
    echo "No destination dir provided."
    echo "Example usage: $0 /tmp/tfs_api_pkg"
    exit 1
  fi

  if [[ ! -d "bazel-bin/tensorflow_serving" ]]; then
    echo "Could not find bazel-bin. Did you run from the root of the build "\
      "tree?"
    exit 1
  fi

  DEST="$1"
  TMPDIR="$(mktemp -d)"
  local PIP_SRC_DIR="tensorflow_serving/tools/pip_package"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  mkdir -p ${TMPDIR}/tensorflow_serving/apis

  echo "Adding python files"
  cp bazel-genfiles/tensorflow_serving/apis/*_pb2.py \
    "${TMPDIR}/tensorflow_serving/apis"

  cp bazel-serving/tensorflow_serving/apis/prediction_service_pb2.py \
    "${TMPDIR}/tensorflow_serving/apis"

  touch "${TMPDIR}/tensorflow_serving/apis/__init__.py"
  touch "${TMPDIR}/tensorflow_serving/__init__.py"

  echo "Adding package setup files"
  cp ${PIP_SRC_DIR}/setup.py "${TMPDIR}"

  pushd "${TMPDIR}"
  echo $(date) : "=== Building wheel"
  python setup.py bdist_wheel # >/dev/null
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  rm -rf "${TMPDIR}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
