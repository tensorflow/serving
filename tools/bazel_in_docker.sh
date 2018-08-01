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
#
# Script to run bazel and do builds in a docker container.
#
# Useful to do builds in a hermetic (docker) environment. The script sets
# up the correct user/group names and docker bind mounts to allow build
# a source tree.
#
# Typical usage (to build from lastest upstream source):
# $ git clone https://github.com/tensorflow/serving.git
# $ cd serving
# $ bazel_in_docker.sh bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server
#

set -e

function usage() {
  local progname=$(basename $0)
  echo "Usage:"
  echo "  ${progname} [-d <docker-image-name>] <command> [args ...]"
  echo ""
  echo "Examples:"
  echo "  ${progname} bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server"
  echo "  ${progname} bazel test tensorflow_serving/..."
  echo "  ${progname} -d tensorflow/serving:latest-devel bazel version"
  exit 1
}

function get_switch_user_cmd() {
  local uid=$(id -u)
  local gid=$(id -g)
  local username=$(id -n -u)
  local groupname=$(id -n -g)
  local cmdline="groupadd -f ${groupname} && groupmod -o -g ${gid} ${groupname}"
  cmdline+="; id -u ${username} &>/dev/null || useradd ${username} && usermod -o -u ${uid} -g ${gid} ${username}"
  cmdline+="; chroot --userspec=${username} / "
  echo "${cmdline}"
}

function get_bazel_cmd() {
  echo "cd $(pwd); TEST_TMPDIR=.cache"
}

(( $# < 1 )) && usage
[[ "$1" = "-"* ]] && [[ "$1" != "-d" ]] && usage

IMAGE="tensorflow/serving:nightly-devel"
[[ "$1" = "-d" ]] && IMAGE=$2 && shift 2 || true
[[ "${IMAGE}" = "" ]] && usage

RUN_OPTS=(--rm -it)
if [[ "$1" = "bazel" ]]; then
  CMD="sh -c '$(get_bazel_cmd) $@'"
  RUN_OPTS+=("-v $(pwd):$(pwd)")
else
  CMD="$@"
fi
[[ "${CMD}" = "" ]] && usage
[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command missing from PATH." && usage

echo "== Pulling docker image: ${IMAGE}"
if ! docker pull ${IMAGE} ; then
  echo "WARNING: Failed to docker pull image ${IMAGE}"
fi

echo "== Running cmd: ${CMD}"
docker run ${RUN_OPTS[@]} ${IMAGE} bash -c "$(get_switch_user_cmd) ${CMD}"
