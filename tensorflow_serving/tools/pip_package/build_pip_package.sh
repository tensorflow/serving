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
    echo "Could not find bazel-bin. Did you run from the root of the build"\
      "tree?"
    exit 1
  fi

  local BAZEL_PROJECT_DIR="bazel-${PWD##*/}"
  DEST="$1"
  TMPDIR="$(mktemp -d)"
  local PIP_SRC_DIR="tensorflow_serving/tools/pip_package"

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  mkdir -p ${TMPDIR}/tensorflow_serving/apis
  mkdir -p ${TMPDIR}/tensorflow_serving/config

  BAZEL_OPT_DIR="k8-opt"
  if [[ $(uname -m) == "aarch64" ]]; then
          BAZEL_OPT_DIR="aarch64-opt"
  fi

  echo "Adding python files"
  cp bazel-out/${BAZEL_OPT_DIR}/bin/tensorflow_serving/apis/*_pb2.py \
    "${TMPDIR}/tensorflow_serving/apis"

  cp ${BAZEL_PROJECT_DIR}/tensorflow_serving/apis/*_pb2.py \
    "${TMPDIR}/tensorflow_serving/apis"

  cp ${BAZEL_PROJECT_DIR}/tensorflow_serving/apis/*_grpc.py \
    "${TMPDIR}/tensorflow_serving/apis"

  cp bazel-out/${BAZEL_OPT_DIR}/bin/tensorflow_serving/config/*_pb2.py \
    "${TMPDIR}/tensorflow_serving/config"

  touch "${TMPDIR}/tensorflow_serving/apis/__init__.py"
  touch "${TMPDIR}/tensorflow_serving/config/__init__.py"
  touch "${TMPDIR}/tensorflow_serving/__init__.py"

  echo "Adding package setup files"
  cp ${PIP_SRC_DIR}/setup.py "${TMPDIR}"

  pushd "${TMPDIR}"
  echo $(date) : "=== Building wheel (CPU)"
  python3 setup.py bdist_wheel --universal \
    --project_name tensorflow-serving-api # >/dev/null
  echo $(date) : "=== Building wheel (GPU)"
  python3 setup.py bdist_wheel --universal \
    --project_name tensorflow-serving-api-gpu # >/dev/null
  mkdir -p "${DEST}"
  cp dist/* "${DEST}"
  popd
  rm -rf "${TMPDIR}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
