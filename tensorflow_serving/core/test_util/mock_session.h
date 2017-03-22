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

#ifndef TENSORFLOW_SERVING_TEST_UTIL_MOCK_SESSION_H_
#define TENSORFLOW_SERVING_TEST_UTIL_MOCK_SESSION_H_

#include <gmock/gmock.h>
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A mock of tensorflow::Session.
class MockSession : public tensorflow::Session {
 public:
  MockSession() : Session() {
    ON_CALL(*this, Close()).WillByDefault(::testing::Return(Status::OK()));
  }
  MOCK_METHOD1(Create, ::tensorflow::Status(const GraphDef& graph));
  MOCK_METHOD1(Extend, ::tensorflow::Status(const GraphDef& graph));
  MOCK_METHOD4(Run, ::tensorflow::Status(
                        const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_names,
                        const std::vector<string>& target_nodes,
                        std::vector<Tensor>* outputs));
  MOCK_METHOD6(Run,
               ::tensorflow::Status(
                   const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor>>& inputs,
                   const std::vector<string>& output_names,
                   const std::vector<string>& target_nodes,
                   std::vector<Tensor>* outputs, RunMetadata* run_metadata));
  MOCK_METHOD4(PRunSetup,
               ::tensorflow::Status(const std::vector<string>& input_names,
                                    const std::vector<string>& output_names,
                                    const std::vector<string>& target_nodes,
                                    string* handle));
  MOCK_METHOD4(PRun, ::tensorflow::Status(
                         const string& handle,
                         const std::vector<std::pair<string, Tensor>>& inputs,
                         const std::vector<string>& output_names,
                         std::vector<Tensor>* outputs));
  MOCK_METHOD0(Close, ::tensorflow::Status());
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_TEST_UTIL_MOCK_SESSION_H_
