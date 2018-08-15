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

#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class SavedModelBundleSourceAdapterTest
    : public ::testing::TestWithParam<bool> {
 protected:
  SavedModelBundleSourceAdapterTest() {
    ResourceUtil::Options resource_util_options;
    resource_util_options.devices = {{device_types::kMain, 1}};
    resource_util_ =
        std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));

    ram_resource_ = resource_util_->CreateBoundResource(
        device_types::kMain, resource_kinds::kRamBytes);
    config_.mutable_config()->set_enable_model_warmup(EnableWarmup());
  }

  void TestSavedModelBundleSourceAdapter(const string& export_dir) const {
    std::unique_ptr<Loader> loader;
    {
      std::unique_ptr<SavedModelBundleSourceAdapter> adapter;
      TF_CHECK_OK(SavedModelBundleSourceAdapter::Create(config_, &adapter));
      ServableData<std::unique_ptr<Loader>> loader_data =
          adapter->AdaptOneVersion(
              ServableData<StoragePath>({"", 0}, export_dir));
      TF_ASSERT_OK(loader_data.status());
      loader = loader_data.ConsumeDataOrDie();

      // Let the adapter fall out of scope and be deleted. The loader we got
      // from it should be unaffected. Regression test coverage for b/30202207.
    }

    // We should get a non-empty resource estimate, and we should get the same
    // value twice (via memoization).
    ResourceAllocation first_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&first_resource_estimate));
    EXPECT_FALSE(first_resource_estimate.resource_quantities().empty());
    ResourceAllocation second_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&second_resource_estimate));
    EXPECT_THAT(second_resource_estimate, EqualsProto(first_resource_estimate));

    TF_ASSERT_OK(loader->Load());

    // We should get a new (lower) resource estimate post-load.
    ResourceAllocation expected_post_load_resource_estimate =
        first_resource_estimate;
    resource_util_->SetQuantity(
        ram_resource_,
        resource_util_->GetQuantity(ram_resource_, first_resource_estimate) -
            config_.config().experimental_transient_ram_bytes_during_load(),
        &expected_post_load_resource_estimate);
    ResourceAllocation actual_post_load_resource_estimate;
    TF_ASSERT_OK(
        loader->EstimateResources(&actual_post_load_resource_estimate));
    EXPECT_THAT(actual_post_load_resource_estimate,
                EqualsProto(expected_post_load_resource_estimate));

    const SavedModelBundle* bundle = loader->servable().get<SavedModelBundle>();
    test_util::TestSingleRequest(bundle->session.get());

    loader->Unload();
  }

  bool EnableWarmup() { return GetParam(); }

  std::unique_ptr<ResourceUtil> resource_util_;
  Resource ram_resource_;
  SessionBundleSourceAdapterConfig config_;
};

TEST_P(SavedModelBundleSourceAdapterTest, Basic) {
  config_.mutable_config()->set_experimental_transient_ram_bytes_during_load(
      42);

  TestSavedModelBundleSourceAdapter(test_util::GetTestSavedModelPath());
}

TEST_P(SavedModelBundleSourceAdapterTest, BackwardCompatibility) {
  TestSavedModelBundleSourceAdapter(
      test_util::GetTestSessionBundleExportPath());
}

// Test all SavedModelBundleSourceAdapterTest test cases with
// warmup enabled/disabled.
INSTANTIATE_TEST_CASE_P(EnableWarmup, SavedModelBundleSourceAdapterTest,
                        ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
