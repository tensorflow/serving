#!/bin/bash -e
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Builds protobuf 3 and grpc for Linux inside the local build tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRPC_SOURCE="${SCRIPT_DIR}/downloads/grpc"
GRPC_GENDIR="${SCRIPT_DIR}/gen/grpc"
PROTOBUF_SOURCE="${GRPC_SOURCE}/third_party/protobuf"
PROTOBUF_GENDIR="${GRPC_GENDIR}"
mkdir -p "${GRPC_GENDIR}"

if [[ ! -f "${PROTOBUF_SOURCE}/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "/contrib/makefile_client/download_dependencies.sh" 1>&2
    exit 1
fi

source "${SCRIPT_DIR}"/build_helper.subr
JOB_COUNT="${JOB_COUNT:-$(get_job_count)}"

# Compile and install protobuf locally
cd ${PROTOBUF_SOURCE}

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

./configure --prefix="${PROTOBUF_GENDIR}" --with-pic
if [ $? -ne 0 ]
then
  echo "./configure command failed."
  exit 1
fi

make clean

make -j"${JOB_COUNT}"
if [ $? -ne 0 ]
then
  echo "make command failed."
  exit 1
fi

make install

# Compile and install grpc locally 
cd ${GRPC_SOURCE}

prefix=${GRPC_GENDIR} make 
prefix=${GRPC_GENDIR} make  install

echo "$(basename $0) finished successfully!!!"
