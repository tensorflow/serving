/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.h"

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

Loader::Metadata CreateMetadata() { return {ServableId{"name", 42}}; }

class TfrtSavedModelSourceAdapterTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));
  }

  TfrtSavedModelSourceAdapterTest() {
    ResourceUtil::Options resource_util_options;
    resource_util_options.devices = {{device_types::kMain, 1}};
    resource_util_ =
        std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));

    ram_resource_ = resource_util_->CreateBoundResource(
        device_types::kMain, resource_kinds::kRamBytes);
    config_.mutable_saved_model_config()
        ->mutable_legacy_config()
        ->set_enable_model_warmup(EnableWarmup());
    if (EnableNumRequestIterations()) {
      config_.mutable_saved_model_config()
          ->mutable_legacy_config()
          ->mutable_model_warmup_options()
          ->mutable_num_request_iterations()
          ->set_value(2);
    }

    config_.mutable_saved_model_config()
        ->mutable_legacy_config()
        ->set_enable_session_metadata(true);
  }

  void TestTFRTSavedModelSourceAdapter(const string& export_dir) const {
    std::unique_ptr<Loader> loader;
    {
      std::unique_ptr<TfrtSavedModelSourceAdapter> adapter;
      TF_CHECK_OK(TfrtSavedModelSourceAdapter::Create(config_, &adapter));
      ServableData<std::unique_ptr<Loader>> loader_data =
          adapter->AdaptOneVersion(
              ServableData<StoragePath>({"", 0}, export_dir));
      TF_ASSERT_OK(loader_data.status());
      loader = loader_data.ConsumeDataOrDie();

      // Let the adapter fall out of scope and be deleted. The loader we got
      // from it should be unaffected.
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
        resource_util_->GetQuantity(ram_resource_, first_resource_estimate),
        &expected_post_load_resource_estimate);
    ResourceAllocation actual_post_load_resource_estimate;
    TF_ASSERT_OK(
        loader->EstimateResources(&actual_post_load_resource_estimate));
    EXPECT_THAT(actual_post_load_resource_estimate,
                EqualsProto(expected_post_load_resource_estimate));

    tfrt::SavedModel& saved_model =
        down_cast<TfrtSavedModelServable*>(loader->servable().get<Servable>())
            ->saved_model();
    TestSingleRequest(&saved_model);

    loader->Unload();
  }

  void TestSingleRequest(tfrt::SavedModel* saved_model) const {
    Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    // half plus two: output should be input / 2 + 2.
    Tensor expected_output =
        test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});

    std::vector<tensorflow::Tensor> input_tensors;
    input_tensors.push_back(input);
    tfrt::SavedModel::RunOptions run_options;
    std::vector<tensorflow::Tensor> output_tensors;
    TF_ASSERT_OK(saved_model->Run(run_options, "serving_default", input_tensors,
                                  &output_tensors));

    ASSERT_EQ(1, output_tensors.size());
    const auto& single_output = output_tensors.at(0);
    test::ExpectTensorEqual<float>(expected_output, single_output);
  }

  bool EnableWarmup() const { return std::get<0>(GetParam()); }
  bool EnableNumRequestIterations() const { return std::get<1>(GetParam()); }

  std::unique_ptr<ResourceUtil> resource_util_;
  Resource ram_resource_;
  TfrtSavedModelSourceAdapterConfig config_;
};

TEST_P(TfrtSavedModelSourceAdapterTest, Basic) {
  TestTFRTSavedModelSourceAdapter(
      test_util::TestSrcDirPath("servables/tensorflow/testdata/"
                                "saved_model_half_plus_two_cpu/00000123"));
}

TEST_P(TfrtSavedModelSourceAdapterTest, MLMetadata) {
  TestTFRTSavedModelSourceAdapter(
      test_util::TestSrcDirPath("servables/tensorflow/testdata/"
                                "saved_model_half_plus_two_mlmd/00000123"));
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
// warmup and num_request_iterations enabled/disabled.
INSTANTIATE_TEST_CASE_P(VariousOptions, TfrtSavedModelSourceAdapterTest,
                        ::testing::Combine(::testing::Bool(),
                                           ::testing::Bool()));

}  // namespace
}  // namespace serving
}  // namespace tensorflow
