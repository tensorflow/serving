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

#ifndef TENSORFLOW_SERVING_TEST_UTIL_TEST_UTIL_H_
#define TENSORFLOW_SERVING_TEST_UTIL_TEST_UTIL_H_

#include <string>

#include "google/protobuf/text_format.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Creates a proto message of type T from a textual representation.
template <typename T>
T CreateProto(const string& textual_proto);

// Creates an absolute test srcdir path to the linked in runfiles given a path
// relative to the current workspace.
// e.g. relative path = "tensorflow_serving/session_bundle".
string TestSrcDirPath(const string& relative_path);

//////////
// Implementation details. API readers need not read.

template <typename T>
T CreateProto(const string& textual_proto) {
  T proto;
  CHECK(protobuf::TextFormat::ParseFromString(textual_proto, &proto));
  return proto;
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_TEST_UTIL_TEST_UTIL_H_
