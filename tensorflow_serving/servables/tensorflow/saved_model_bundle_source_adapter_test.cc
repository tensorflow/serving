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

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/test_util/session_test_util.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/oss_or_google.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

Loader::Metadata CreateMetadata() { return {ServableId{"name", 42}}; }

class SavedModelBundleSourceAdapterTest
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  SavedModelBundleSourceAdapterTest() {
    ResourceUtil::Options resource_util_options;
    resource_util_options.devices = {{device_types::kMain, 1}};
    resource_util_ =
        std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));

    ram_resource_ = resource_util_->CreateBoundResource(
        device_types::kMain, resource_kinds::kRamBytes);
    config_.mutable_legacy_config()->set_enable_model_warmup(EnableWarmup());
    if (EnableNumRequestIterations()) {
      config_.mutable_legacy_config()
          ->mutable_model_warmup_options()
          ->mutable_num_request_iterations()
          ->set_value(2);
    }

    config_.mutable_legacy_config()->set_enable_session_metadata(
        EnableSessionMetadata());

    config_.mutable_legacy_config()->set_session_target(
        test_util::kNewSessionHookSessionTargetPrefix);
    test_util::SetNewSessionHook([&](const SessionOptions& session_options) {
      EXPECT_EQ(EnableSessionMetadata(),
                session_options.config.experimental().has_session_metadata());
      if (EnableSessionMetadata()) {
        const auto& actual_session_metadata =
            session_options.config.experimental().session_metadata();
        const auto& expected_loader_metadata = CreateMetadata();
        EXPECT_EQ(expected_loader_metadata.servable_id.name,
                  actual_session_metadata.name());
        EXPECT_EQ(expected_loader_metadata.servable_id.version,
                  actual_session_metadata.version());
      }
      return OkStatus();
    });
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

    const auto metadata = CreateMetadata();
    TF_ASSERT_OK(loader->LoadWithMetadata(CreateMetadata()));

    // We should get a new (lower) resource estimate post-load.
    ResourceAllocation expected_post_load_resource_estimate =
        first_resource_estimate;
    resource_util_->SetQuantity(
        ram_resource_,
        resource_util_->GetQuantity(ram_resource_, first_resource_estimate) -
            config_.legacy_config()
                .experimental_transient_ram_bytes_during_load(),
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

  bool EnableWarmup() const { return std::get<0>(GetParam()); }
  bool EnableNumRequestIterations() const { return std::get<1>(GetParam()); }
  bool EnableSessionMetadata() const { return std::get<2>(GetParam()); }

  std::unique_ptr<ResourceUtil> resource_util_;
  Resource ram_resource_;
  SavedModelBundleSourceAdapterConfig config_;
};

TEST_P(SavedModelBundleSourceAdapterTest, Basic) {
  config_.mutable_legacy_config()
      ->set_experimental_transient_ram_bytes_during_load(42);

  TestSavedModelBundleSourceAdapter(test_util::GetTestSavedModelPath());
}

TEST_P(SavedModelBundleSourceAdapterTest, BackwardCompatibility) {
  if (IsTensorflowServingOSS()) {
    return;
  }
  TestSavedModelBundleSourceAdapter(
      test_util::GetTestSessionBundleExportPath());
}

TEST_P(SavedModelBundleSourceAdapterTest, MLMetadata) {
  if (!EnableSessionMetadata()) return;
  TestSavedModelBundleSourceAdapter(test_util::TestSrcDirPath(
      strings::StrCat("/servables/tensorflow/testdata/",
                      "saved_model_half_plus_two_mlmd/00000123")));
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics(options);
  const monitoring::PointSet& lps =
      *collected_metrics->point_set_map.at("/tensorflow/serving/mlmd_map");

  EXPECT_EQ(1, lps.points.size());
  EXPECT_EQ(2, lps.points[0]->labels.size());
  EXPECT_EQ("model_name", lps.points[0]->labels[0].name);
  EXPECT_EQ("name", lps.points[0]->labels[0].value);
  EXPECT_EQ("version", lps.points[0]->labels[1].name);
  EXPECT_EQ("42", lps.points[0]->labels[1].value);
  EXPECT_EQ("test_mlmd_uuid", lps.points[0]->string_value);
}

// Test all SavedModelBundleSourceAdapterTest test cases with
// warmup, num_request_iterations enabled/disabled and session-metadata
// enabled/disabled.
INSTANTIATE_TEST_CASE_P(VariousOptions, SavedModelBundleSourceAdapterTest,
                        ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                           ::testing::Bool()));

}  // namespace
}  // namespace serving
}  // namespace tensorflow
