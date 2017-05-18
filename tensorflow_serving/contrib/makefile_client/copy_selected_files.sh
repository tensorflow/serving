#!/bin/bash -e
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Copies files needed to build Tensorflow Serving clients to private sandbox. 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE_ROOT_DIR="../../../"
SOURCE_CC_FILE_LIST_NAME="cc_client_cc_source_files.txt"
SOURCE_H_FILE_LIST_NAME="cc_client_h_source_files.txt"
SOURCE_PROTO_FILE_LIST_NAME="cc_client_proto_source_files.txt"

SOURCE_CC_FILE_LIST="${SCRIPT_DIR}/${SOURCE_CC_FILE_LIST_NAME}"
SOURCE_H_FILE_LIST="${SCRIPT_DIR}/${SOURCE_H_FILE_LIST_NAME}"
SOURCE_PROTO_FILE_LIST="${SCRIPT_DIR}/${SOURCE_PROTO_FILE_LIST_NAME}"

PROTO_TEXT_UTIL_CC_FILE_LIST_NAME="proto_text_util_cc_files.txt"
PROTO_TEXT_UTIL_H_FILE_LIST_NAME="proto_text_util_h_files.txt"
PROTO_TEXT_UTIL_PROTO_FILE_LIST_NAME="proto_text_util_proto_files.txt"

PROTO_TEXT_UTIL_CC_FILE_LIST="${SCRIPT_DIR}/${PROTO_TEXT_UTIL_CC_FILE_LIST_NAME}"
PROTO_TEXT_UTIL_H_FILE_LIST="${SCRIPT_DIR}/${PROTO_TEXT_UTIL_H_FILE_LIST_NAME}"
PROTO_TEXT_UTIL_PROTO_FILE_LIST="${SCRIPT_DIR}/${PROTO_TEXT_UTIL_PROTO_FILE_LIST_NAME}"

SOURCE_SLICE_DIR='code-slice'
SOURCE_SLICE_TEMP_DIR='code-slice/temp'

mkdir -p "${SOURCE_SLICE_DIR}"
mkdir -p "${SOURCE_SLICE_TEMP_DIR}"

if [[ ! -f "${SOURCE_FILE_ROOT_DIR}/tensorflow/configure" ]]; then
    echo "You need to clone the full tensorflow serving git repository to" 1>&2
    echo "be able to copy the needed client files to the client build source code slice." 1>&2
    echo "Or you are running this file copy script from the wrong subdirectory." 1>&2
    exit 1
fi

# TODO(mtm): The --backup=numbered option does not work on the Mac. Add OS specific logic or simply remove it.
# This backup gives a reprieve to those who might edit in place in the code-slice and then copy over it.
echo "Making a backup of old source code slice before overwriting it."
tar cvzf ${SCRIPT_DIR}/old-code-slice-backup.tgz --backup=numbered  ${SOURCE_SLICE_DIR}

echo "* Copying cc source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${SOURCE_CC_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)
echo "* Copying header source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${SOURCE_H_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)
echo "* Copying protobuf source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${SOURCE_PROTO_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)

echo "* Copying proto text util cc source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${PROTO_TEXT_UTIL_CC_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)
echo "* Copying proto text util header source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${PROTO_TEXT_UTIL_H_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)
echo "* Copying proto text util protobuf source files to code-slice."
(cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - -T ${PROTO_TEXT_UTIL_PROTO_FILE_LIST}) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)

# Do special case file setup/copy operations

echo "* Doing special case file setup/copy operations."
echo 'Create "version" file [tensorflow/tensorflow/core/util/version_info.cc] before git is gone.'
${SOURCE_FILE_ROOT_DIR}/tensorflow/tensorflow/tools/git/gen_git_source.sh ${SOURCE_SLICE_TEMP_DIR}/tensorflow/tensorflow/core/util/version_info.cc

echo 'Copy proto_text "place holder" file to code-slice.'
( cd ${SOURCE_FILE_ROOT_DIR} && tar cvf - tensorflow/tensorflow/tools/proto_text/placeholder.txt ) | cat | (cd ${SOURCE_SLICE_TEMP_DIR} && tar xbf 1 -)

# Move source files to normalized locations, so less fiddling needs to be done for protoc includes in Makefile
echo 'Move source files to normalized locations.'

rm -rf ${SOURCE_SLICE_DIR}/tensorflow/
rm -rf ${SOURCE_SLICE_DIR}/tensorflow_serving/

mv  ${SOURCE_SLICE_TEMP_DIR}/tensorflow/tensorflow/ ${SOURCE_SLICE_DIR}/tensorflow/
mv  ${SOURCE_SLICE_TEMP_DIR}/tensorflow_serving/ ${SOURCE_SLICE_DIR}/tensorflow_serving/

rmdir ${SOURCE_SLICE_TEMP_DIR}/tensorflow/
rmdir ${SOURCE_SLICE_TEMP_DIR}

echo "Copy Tensorflow's eigen additions to code-slice"
mkdir ${SOURCE_SLICE_DIR}/third_party

cp -rp ${SOURCE_FILE_ROOT_DIR}/tensorflow/third_party/eigen3 ${SOURCE_SLICE_DIR}/third_party

echo "File copying is done."
