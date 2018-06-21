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
FROM ubuntu:16.04

LABEL maintainer="gvasudevan@google.com"

RUN apt-get update && apt-get install -y \
        curl \
        gnupg

# Install TF Serving pkg.
ARG TF_SERVING_VERSION=1.8.0
# Use tensorflow-model-server-universal for older hardware
ARG TF_SERVING_PKGNAME=tensorflow-model-server
RUN curl -LO https://storage.googleapis.com/tensorflow-serving-apt/pool/${TF_SERVING_PKGNAME}-${TF_SERVING_VERSION}/t/${TF_SERVING_PKGNAME}/${TF_SERVING_PKGNAME}_${TF_SERVING_VERSION}_all.deb ; \
        dpkg -i ${TF_SERVING_PKGNAME}_${TF_SERVING_VERSION}_all.deb ; \
        rm ${TF_SERVING_PKGNAME}_${TF_SERVING_VERSION}_all.deb

# Cleanup to reduce the size of the image
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN  apt-get clean && \
        rm -rf /var/lib/apt/lists/*

#Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model
ENTRYPOINT tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
