/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/test_util/test_util.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace test_util {

string TestSrcDirPath(const string& relative_path) {
  return tensorflow::io::JoinPath(
      tensorflow::io::JoinPath(getenv("TEST_SRCDIR"),
                               "tensorflow_serving"),
      relative_path);
}

ProtoStringMatcher::ProtoStringMatcher(const string& expected)
    : expected_(expected) {}
ProtoStringMatcher::ProtoStringMatcher(const google::protobuf::Message& expected)
    : expected_(expected.DebugString()) {}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
