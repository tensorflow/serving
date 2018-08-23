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
FROM quay.io/kairosinc/serving-cpu:0.0.2 as build_env

COPY . /tensorflow-serving
# Download TF Serving sources (optionally at specific commit).
WORKDIR /tensorflow-serving

# Build, and install TensorFlow Serving
ARG TF_SERVING_BUILD_OPTIONS="--copt=-mavx --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"
# use "--verbose_failures --local_resources=4096,1.0,1.0" for local build
ARG TF_SERVING_BAZEL_OPTIONS="--verbose_failures --local_resources=4096,1.0,1.0"
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    bazel build -c opt --color=yes --curses=yes --config=cuda \
    ${TF_SERVING_BAZEL_OPTIONS} \
    --output_filter=DONT_MATCH_ANYTHING \
    ${TF_SERVING_BUILD_OPTIONS} \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel clean --expunge --color=yes
# Clean up Bazel cache when done.

FROM nvidia/cuda:9.0-base-ubuntu16.04

# Install dependencies
RUN apt-get update && apt-get install -y \
        curl \
        libcurl3-dev \
        zip \
        unzip \
        jq \
        gcc \
        software-properties-common \
        python-dev \
        python-pip \
        && apt-get clean \
        && apt-get autoremove \
        && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
        && apt-get update \
        && apt-get install -y libstdc++6 \
        && pip install awscli


# Add scripts, in future liveness and readiness scripts too
COPY scripts scripts
COPY --from=build_env /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server
# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models 
# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model
# if not set, will not sync with s3; e.g.: "s3://mybucket"
ENV S3_BUCKET="" 
RUN \
    mkdir -p ${MODEL_BASE_PATH} \
    && chmod +x /usr/bin/tensorflow_model_server scripts/entrypoint.sh \
    && ls -la /usr/bin/tensorflow_model_server

EXPOSE 8500 # gRPC
EXPOSE 8501 # REST
ENTRYPOINT scripts/entrypoint.sh