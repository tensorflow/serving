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

#include "tensorflow_serving/servables/tensorflow/simple_servers.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

class SimpleServersTest : public ::testing::Test {
 protected:
  SimpleServersTest()
      : test_data_path_(test_util::TensorflowTestSrcDirPath(
            "cc/saved_model/testdata/half_plus_two")) {}

  // Test that a SavedModelBundle handles a single request for the half plus two
  // model properly. The request has size=2, for batching purposes.
  void TestSingleRequest(const SavedModelBundle& bundle) {
    const Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    // half plus two: output should be input / 2 + 2.
    const Tensor expected_output =
        test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});

    // Note that "x" and "y" are the actual names of the nodes in the graph.
    // The saved manifest binds these to "input" and "output" respectively, but
    // these tests are focused on the raw underlying session without bindings.
    const std::vector<std::pair<string, Tensor>> inputs = {{"x", input}};
    const std::vector<string> output_names = {"y"};
    const std::vector<string> empty_targets;
    std::vector<Tensor> outputs;

    TF_ASSERT_OK(
        bundle.session->Run(inputs, output_names, empty_targets, &outputs));

    ASSERT_EQ(1, outputs.size());
    const auto& single_output = outputs.at(0);
    test::ExpectTensorEqual<float>(expected_output, single_output);
  }

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string test_data_path_;
};

TEST_F(SimpleServersTest, Basic) {
  std::unique_ptr<Manager> manager;
  const Status status = simple_servers::CreateSingleTFModelManagerFromBasePath(
      test_data_path_, &manager);
  TF_CHECK_OK(status);
  // We wait until the manager starts serving the servable.
  // TODO(b/25545570): Use the waiter api when it's ready.
  while (manager->ListAvailableServableIds().empty()) {
    Env::Default()->SleepForMicroseconds(1000);
  }
  ServableHandle<SavedModelBundle> bundle;
  const Status handle_status =
      manager->GetServableHandle(ServableRequest::Latest("default"), &bundle);
  TF_CHECK_OK(handle_status);
  TestSingleRequest(*bundle);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
