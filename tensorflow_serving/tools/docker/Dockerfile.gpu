# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel-gpu

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM nvidia/cuda:10.0-base-ubuntu16.04

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL maintainer="gvasudevan@google.com"
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

ENV CUDNN_VERSION=7.4.1.5
ENV TF_TENSORRT_VERSION=5.0.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-command-line-tools-10-0 \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=${CUDNN_VERSION}-1+cuda10.0 \
        libgomp1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The 'apt-get install' of nvinfer-runtime-trt-repo-* library adds a new list
# which contains libnvinfer library, so it needs another 'apt-get update' to
# retrieve that list before it can actually install the library.
#
# We don't install libnvinfer-dev since we don't need to build against TensorRT.
RUN apt-get update && \
    apt-get install --no-install-recommends \
        nvinfer-runtime-trt-repo-ubuntu1604-${TF_TENSORRT_VERSION}-ga-cuda10.0 && \
    apt-get update && \
    apt-get install --no-install-recommends \
        libnvinfer5=${TF_TENSORRT_VERSION}-1+cuda10.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm /usr/lib/x86_64-linux-gnu/libnvcaffe_parser* && \
    rm /usr/lib/x86_64-linux-gnu/libnvparsers*

# Install TF Serving GPU pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
