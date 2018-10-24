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
# Tests if a Docker image built from Dockerfile.gpu basically functions.
#
# It does this by loading up a half plus two toy model in the Docker image
# and querying it, validating the response. This model will only load on a GPU.
#
# The image passed to this test must be already available locally.
#
# Ex: $ bazel test :unittest_Dockerfile.gpu \
#         --test_arg=tensorflow/serving:latest-gpu \
#         --test_output=streamed --verbose_failures

declare -r PROJDIR=$(pwd)/tensorflow_serving
source ${PROJDIR}/tools/docker/tests/docker_test_lib.sh || exit 1

# Values to fill in for test
# ------------------------------------------------------------------------------
declare -r USE_NVIDIA_RUNTIME=true
declare -r IS_MKL_IMAGE=false
declare -r IS_DEVEL_IMAGE=false
declare -r MODELNAME="saved_model_half_plus_two_gpu"
declare -r MODELDIR="${PROJDIR}/servables/tensorflow/testdata"
declare -r REQUEST='{"instances": [1.0,2.0,5.0]}'
declare -r RESPONSE='{"predictions":[2.5,3.0,4.5]}'
# ------------------------------------------------------------------------------

# Grab the last argument as the image, so we can override the test arg in
# the BUILD file
test_docker_image ${@: -1}
