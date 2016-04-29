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

#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/test_util/source_adapter_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class SessionBundleSourceAdapterTest : public ::testing::Test {
 protected:
  SessionBundleSourceAdapterTest()
      : export_dir_(test_util::TestSrcDirPath(
            "session_bundle/example/half_plus_two/00000123")) {}

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;

  // Test that a SessionBundle handles a single request for the half plus two
  // model properly. The request has size=2, for batching purposes.
  void TestSingleRequest(const SessionBundle* bundle) {
    Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    // half plus two: output should be input / 2 + 2.
    Tensor expected_output =
        test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});

    // Note that "x" and "y" are the actual names of the nodes in the graph.
    // The saved manifest binds these to "input" and "output" respectively, but
    // these tests are focused on the raw underlying session without bindings.
    const std::vector<std::pair<string, Tensor>> inputs = {{"x", input}};
    const std::vector<string> output_names = {"y"};
    const std::vector<string> empty_targets;
    std::vector<Tensor> outputs;

    TF_ASSERT_OK(
        bundle->session->Run(inputs, output_names, empty_targets, &outputs));

    ASSERT_EQ(1, outputs.size());
    const auto& single_output = outputs.at(0);
    test::ExpectTensorEqual<float>(expected_output, single_output);
  }

  void TestSessionBundleSourceAdapter(
      const SessionBundleSourceAdapterConfig& config) {
    std::unique_ptr<SessionBundleSourceAdapter> adapter;
    TF_CHECK_OK(SessionBundleSourceAdapter::Create(config, &adapter));
    ServableData<std::unique_ptr<Loader>> loader_data =
        test_util::RunSourceAdapter(export_dir_, adapter.get());
    TF_ASSERT_OK(loader_data.status());
    std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();

    // We should get a non-empty resource estimate, and we should get the same
    // value twice (via memoization).
    ResourceAllocation first_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&first_resource_estimate));
    EXPECT_FALSE(first_resource_estimate.resource_quantities().empty());
    ResourceAllocation second_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&second_resource_estimate));
    EXPECT_THAT(second_resource_estimate, EqualsProto(first_resource_estimate));

    TF_ASSERT_OK(loader->Load(ResourceAllocation()));

    const SessionBundle* bundle = loader->servable().get<SessionBundle>();
    TestSingleRequest(bundle);

    loader->Unload();
  }
};

TEST_F(SessionBundleSourceAdapterTest, Basic) {
  const SessionBundleSourceAdapterConfig config;
  TestSessionBundleSourceAdapter(config);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
