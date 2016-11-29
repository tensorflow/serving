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

#include "tensorflow_serving/servables/caffe/caffe_source_adapter.h"

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
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/test_util/source_adapter_test_util.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_config.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_source_adapter.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class CaffeSourceAdapterTest : public ::testing::Test {
 protected:
  CaffeSourceAdapterTest()
      : export_dir_(test_util::TestSrcDirPath(
            "servables/caffe/test_data/mnist_pretrained_caffe/00000023")) {}

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;

  void TestSessionBundleSourceAdapter(const CaffeSourceAdapterConfig& config) {
    std::unique_ptr<CaffeSourceAdapter> adapter;
    TF_CHECK_OK(CaffeSourceAdapter::Create(config, &adapter));

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

    const CaffeSessionBundle* bundle =
        loader->servable().get<CaffeSessionBundle>();
    ASSERT_EQ(bundle->meta_graph_def.resolved_inputs.size(), 1);
    ASSERT_EQ(bundle->meta_graph_def.resolved_inputs[0], "images");

    ASSERT_EQ(bundle->meta_graph_def.resolved_outputs.size(), 1);
    ASSERT_EQ(bundle->meta_graph_def.resolved_outputs[0], "scores");

    ASSERT_EQ(bundle->meta_graph_def.classes.dtype(), DT_STRING);
    ASSERT_EQ(bundle->meta_graph_def.classes.string_val().size(), 10);

    loader->Unload();
  }
};

TEST_F(CaffeSourceAdapterTest, Basic) {
  const CaffeSourceAdapterConfig config;
  TestSessionBundleSourceAdapter(config);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
