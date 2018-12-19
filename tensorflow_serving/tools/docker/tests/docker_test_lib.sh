#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
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
# Helper functions for unit testing docker images.


declare TMPDIR=""
declare CONTAINER_ID=""
declare IMAGE=""
declare MODEL_FULL_PATH=""

# Set PATH since it's cleared in a hermetic environment
function _init_path {
  export PATH=$PATH:/usr/bin/:/bin
}

# Copies model to a temp dir for mounting from Docker
function _copy_modeldir {
  if [[ ! -d ${MODELDIR} ]]; then
    echo "Failed testing image. Missing model dir: ${MODELDIR}"
    exit 1
  fi

  # Copy all data to tmp so it can be mounted by docker
  TMPDIR="$(mktemp -d)"
  cp -Lr ${MODELDIR}/${MODELNAME} ${TMPDIR}
  chmod -R 755 "$TMPDIR"
  MODEL_FULL_PATH=${TMPDIR}"/"${MODELNAME}
}


# Looks for the docker container when an image is spun up
function _find_container {
  local delay=1
  local attempts=0
  local max_attempts=60
  # Wait for container to be loaded (max 60 seconds).
  while (( attempts++ < max_attempts )) && [[ -z ${CONTAINER_ID} ]]; do
    sleep $delay
    echo "Polling for container ID for image ${IMAGE}..."
    CONTAINER_ID=$(docker ps | grep "${IMAGE}" | awk '{print $1}')
  done
  if (( attempts > max_attempts )); then
    echo "Failed to get container ID for image: ${IMAGE}."
    echo "Running docker containers:"
    docker ps
    exit 1
  fi
  echo "Container ${CONTAINER_ID} Details:"
  docker inspect ${CONTAINER_ID}
}

# Waits for the model server to indicate that its HTTP endpoint is up
function _wait_for_http {
  # Wait for HTTP end-point to be up
  local delay=5
  local attempts=0
  local max_attempts=60
  # May need to wait up to 5 minutes for the GPU build
  echo "Waiting for HTTP endpoint"
  while (( attempts++ < max_attempts )); do
    sleep $delay
    echo "Polling for HTTP endpoint for image ${IMAGE}..."
    if [[ ! -z $(docker logs ${CONTAINER_ID} | grep "Exporting HTTP/REST API") ]]; then
      break
    fi
  done
  if (( attempts > max_attempts )); then
    echo "HTTP endpoint never came up for image: ${IMAGE}."
    echo "Docker logs:"
    docker logs ${CONTAINER_ID}
    exit 1
  fi
}

# Queries the model server to validate the docker image
function _query_model {
  # Try querying model few times.
  local delay=5
  local attempts=0
  local max_attempts=3
  echo "Will query ModelServer for $max_attempts (max) attempts..."
  while (( attempts++ < max_attempts )); do
    echo "Querying TF ModelServer (attempt: ${attempts})"
    result=$(curl -s -d "${REQUEST}" \
      -X POST http://localhost:8501/v1/models/${MODELNAME}:predict)
    if echo "$result" | tr -d '\n ' \
      | grep -q -s -F "${RESPONSE}"; then
      break
    fi
    echo -e "Unexpected response:\n${result}"

    if (( attempts < max_attempts )); then
      echo "Sleeping for $delay seconds and retrying again..."
      sleep $delay
    fi
  done
  if (( attempts > max_attempts )); then
    echo "Failed to query model."
    exit 1
  fi
  echo "Successfully queries model. Response:\n${result}"
}

# Cleans up test artifacts
function _cleanup_test {
  echo "Stopping TF ModelServer"
  docker stop -t0 ${CONTAINER_ID}

  echo "Clean up TMPDIR:${TMPDIR}"
  [[ -z ${TMPDIR} ]] || rm -rf ${TMPDIR}
}

# Main test function
# Args:
#   $1: Docker image name
function test_docker_image {
  _init_path

  IMAGE="$1"
  _copy_modeldir

  local rest_port="8501"
  local model_base_path="/models/${MODELNAME}"
  local docker_opts=" --privileged=true --rm -t -p ${rest_port}:${rest_port}"
  docker_opts+=" -v ${MODEL_FULL_PATH}:${model_base_path}"
  if [ "$USE_NVIDIA_RUNTIME" = true ]; then
      docker_opts+=" --runtime=nvidia"
  fi

  if [[ "$IS_MKL_IMAGE" = true ]]; then
      docker_opts+=" -e MKLDNN_VERBOSE=1"
  fi

  if [ -z $(docker images -q ${IMAGE}) ]; then
    echo "Docker image ${IMAGE} doesn't exist, please create or pull"
    exit 1
  fi

  echo "Starting TF ModelServer in Docker container from image:${IMAGE} ..."
  if [[ "$IS_DEVEL_IMAGE" = true ]]; then
    echo "Starting TF ModelServer in Docker container from image:${image} ..."
    # Devel images do not run ModelServer but rather start interative shell.
    # Hence we need to explicitly set entrypoint to modelserver and pass the
    # required commandline options to serve the model.
    docker run ${docker_opts} \
      --entrypoint /usr/local/bin/tensorflow_model_server -- ${IMAGE} \
      --rest_api_port=${rest_port} \
      --model_name=${MODELNAME} --model_base_path=${model_base_path} &
  else
    docker run ${docker_opts} -e MODEL_NAME=${MODELNAME} ${IMAGE} &
  fi
  _find_container
  _wait_for_http
  _query_model

  if [[ "$IS_MKL_IMAGE" = true ]]; then
    echo "Checking for mkldnn_verbose in logs of container: ${CONTAINER_ID}"
    if [[ -z $(docker logs --tail 1 ${CONTAINER_ID} | grep "mkldnn_verbose") ]]; then
        echo "${IMAGE}: does not use MKL optimizations"
        exit 1
    fi
  fi

  _cleanup_test

}
