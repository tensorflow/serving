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

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Creates a proto message of type T from a textual representation.
template <typename T>
T CreateProto(const string& textual_proto);

// Return an absolute runfiles srcdir given a path relative to
// tensorflow.
string TensorflowTestSrcDirPath(const string& relative_path);

// Return an absolute runfiles srcdir given a path relative to
// tensorflow/contrib.
string ContribTestSrcDirPath(const string& relative_path);

// Return an absolute runfiles srcdir given a path relative to
// tensorflow_serving.
string TestSrcDirPath(const string& relative_path);

// Simple implementation of a proto matcher comparing string representations.
//
// IMPORTANT: Only use this for protos whose textual representation is
// deterministic (that may not be the case for the map collection type).
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const string& expected);
  explicit ProtoStringMatcher(const google::protobuf::Message& expected);

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener* /* listener */) const;

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const string expected_;
};

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const google::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

//////////
// Implementation details. API readers need not read.

template <typename T>
T CreateProto(const string& textual_proto) {
  T proto;
  CHECK(protobuf::TextFormat::ParseFromString(textual_proto, &proto));
  return proto;
}

template <typename Message>
bool ProtoStringMatcher::MatchAndExplain(
    const Message& p, ::testing::MatchResultListener* /* listener */) const {
  // Need to CreateProto and then print as string so that the formatting
  // matches exactly.
  return p.SerializeAsString() ==
         CreateProto<Message>(expected_).SerializeAsString();
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_TEST_UTIL_TEST_UTIL_H_
