#!/bin/bash -x
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
# This script generates the source file lists needed by the makefile by querying
# the master Bazel build configuration.

bazel query 'kind("source file", deps(//tensorflow_serving/example:inception_client_cc))' | \
grep "//tensorflow.*/.*\.proto$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> cc_client_proto_source_files.txt

bazel query 'kind("source file", deps(//tensorflow_serving/example:inception_client_cc))' | \
grep "//tensorflow.*/.*\.cc$" | \
grep -v "platform/windows" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> cc_client_cc_source_files.txt

bazel query 'kind("source file", deps(//tensorflow_serving/example:inception_client_cc))' | \
grep "//tensorflow.*/.*\.h$" | \
grep -v "platform/windows" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> cc_client_h_source_files.txt

bazel query 'kind("generated file", deps(//tensorflow_serving/example:inception_client_cc))' | \
grep "pb_text\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> cc_client_pb_text_cc_generated_files.txt

bazel query 'kind("generated file", deps(//tensorflow_serving/example:inception_client_cc))' | \
grep "grpc.pb\.cc$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#\.grpc\.pb\.cc#\.proto#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> cc_client_grpc_proto_generated_files.txt

## Generate list of files needed to build proto_text_functions utility

bazel query 'kind("source file", deps(@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
grep -v "platform/windows" | \
grep -v "jpeg" | \
grep -v "png" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> proto_text_util_cc_files.txt

bazel query 'kind("source file", deps(@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.h$" | \
grep -v "platform/windows" | \
grep -v "jpeg" | \
grep -v "png" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> proto_text_util_h_files.txt

bazel query 'kind("source file", deps(@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.proto$" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> proto_text_util_proto_files.txt

bazel query 'kind("generated file", deps(@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.cc$" | \
grep -v "platform/windows" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> proto_text_util_pb_cc_files.txt

bazel query 'kind("generated file", deps(@org_tensorflow//tensorflow/tools/proto_text:gen_proto_text_functions))' | \
grep -E "//tensorflow/.*\.h$" | \
grep -v "platform/windows" | \
sed -E 's#^//##g' | \
sed -E 's#:#/#g' | \
sed -E 's#@org_tensorflow//#tensorflow/#g' \
> proto_text_util_pb_h_files.txt

