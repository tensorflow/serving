# This file is based on Google TensorFlow Serving Dockerfile file with
# modifications needed to build the project for CentOS7.
# Original file is licenced under Apache Licence, Version 2.0:
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile

ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tf-serving-centos7-devel:${TF_SERVING_VERSION}

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM centos:7.4.1708

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN yum update -y && yum install -y ca-certificates && yum clean all && rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
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
# COPY tf_serving_entrypoint.sh /usr/bin/tf_serving_entrypoint.sh

RUN echo '#!/bin/bash' > /usr/bin/tf_serving_entrypoint.sh \
    && echo '' >> /usr/bin/tf_serving_entrypoint.sh\
    && echo '' >> /usr/bin/tf_serving_entrypoint.sh\
    && echo  'tensorflow_model_server --port=8500 --rest_api_port=8501 \
            --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
            "$@"' >> /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh


ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
