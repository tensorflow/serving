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
# Script to run commands (like builds and scripts) in a docker container.
#
# Useful to do builds in a hermetic (docker) environment. The script sets
# up the correct user/group names and docker bind mounts to allow build
# a source tree. Also useful for running python scripts without having to
# track down dependencies (that are covered by the docker container).
#
# Note: This script binds your working directory (via pwd) and /tmp to the
# Docker container. Any scripts or programs you run will need to have its
# output files/dirs written to one of the above locations for persistence.
#
# Typical usage (to build from lastest upstream source):
# $ git clone https://github.com/tensorflow/serving.git
# $ cd serving
# $ ./tools/run_in_docker.sh bazel build tensorflow_serving/model_servers:tensorflow_model_server
#
# Running a python script:
# $ cd serving
# $ ./tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist

set -e

function usage() {
  local progname=$(basename $0)
  echo "Usage:"
  echo "  ${progname} [-d <docker-image-name>] [-o <docker-run-options>] <command> [args ...]"
  echo ""
  echo "Examples:"
  echo "  ${progname} bazel build tensorflow_serving/model_servers:tensorflow_model_server"
  echo "  ${progname} python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist"
  echo "  ${progname} -d tensorflow/serving:latest-devel bazel version"
  exit 1
}

function get_switch_user_cmd() {
  local uid=$(id -u)
  local gid=$(id -g)
  local username=$(id -n -u)
  local groupname=$(id -n -g)
  local cmdline="groupadd -f ${groupname} && groupmod -o -g ${gid} ${groupname}"
  cmdline+="; id -u ${username} &>/dev/null || useradd -N ${username} && usermod -o -u ${uid} -g ${gid} ${username}"
  cmdline+="; chroot --userspec=${username} / "
  echo "${cmdline}"
}

function get_bazel_cmd() {
  echo "cd $(pwd); TEST_TMPDIR=.cache"
}

function get_python_cmd() {
  echo "cd $(pwd);"
}

(( $# < 1 )) && usage

IMAGE="tensorflow/serving:nightly-devel"
RUN_OPTS=()
while [[ $# > 1 ]]; do
  case "$1" in
    -d)
      IMAGE="$2"; shift 2;;
    -o)
      RUN_OPTS=($2); shift 2;;
    *)
      break;;
  esac
done

RUN_OPTS+=(--rm -it --network=host)
# Map the working directory and /tmp to allow scripts/binaries to run and also
# output data that might be used by other scripts/binaries
RUN_OPTS+=("-v $(pwd):$(pwd)")
RUN_OPTS+=("-v /tmp:/tmp")
if [[ "$1" = "bazel"* ]]; then
  CMD="sh -c '$(get_bazel_cmd) $@'"
elif [[ "$1" == "python"* ]]; then
  CMD="sh -c '$(get_python_cmd) $@'"
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
