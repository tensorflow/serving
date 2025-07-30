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

#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_factory.h"

#include <memory>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/session_bundle/graph_rewriter.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

Loader::Metadata CreateMetadata() { return {ServableId{"name", 42}}; }

// Tests TfrtSavedModelFactory with native SavedModel.
class TfrtSavedModelFactoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));
  }
  TfrtSavedModelFactoryTest()
      : model_path_(test_util::TestSrcDirPath(
            "servables/tensorflow/"
            "testdata/saved_model_half_plus_two_cpu/00000123")) {}

  absl::Status CreateTfrtSavedModel(
      const TfrtSavedModelConfig& config,
      std::unique_ptr<tfrt::SavedModel>* saved_model) {
    std::unique_ptr<TfrtSavedModelFactory> factory;
    TF_RETURN_IF_ERROR(TfrtSavedModelFactory::Create(config, &factory));
    TF_RETURN_IF_ERROR(factory->CreateTfrtSavedModelWithMetadata(
        CreateMetadata(), model_path_, saved_model));
    return absl::OkStatus();
  }

  std::vector<string> GetModelFiles() {
    const string& dir = model_path_;
    return {io::JoinPath(dir, kSavedModelAssetsDirectory, "foo.txt"),
            io::JoinPath(dir, kSavedModelFilenamePb),
            io::JoinPath(dir, kSavedModelVariablesFilename,
                         "variables.data-00000-of-00001"),
            io::JoinPath(dir, kSavedModelVariablesFilename, "variables.index")};
  }

  string model_path_;
};

TEST_F(TfrtSavedModelFactoryTest, EstimateResourceRequirementWithGoodExport) {
  TfrtSavedModelConfig config;
  std::unique_ptr<TfrtSavedModelFactory> factory;
  TF_ASSERT_OK(TfrtSavedModelFactory::Create(config, &factory));

  ResourceAllocation actual;
  TF_ASSERT_OK(factory->EstimateResourceRequirement(model_path_, &actual));

  const double total_file_size = test_util::GetTotalFileSize(GetModelFiles());
  ResourceAllocation expected =
      test_util::GetExpectedResourceEstimate(total_file_size);
  EXPECT_THAT(actual, test_util::EqualsProto(expected));
}

TEST_F(TfrtSavedModelFactoryTest, Basic) {
  std::unique_ptr<tfrt::SavedModel> saved_model;
  TfrtSavedModelConfig config;
  *config.add_saved_model_tags() = kSavedModelTagServe;
  TF_ASSERT_OK(CreateTfrtSavedModel(config, &saved_model));

  Tensor input_tensor = test::AsTensor<float>({100.0f, 42.0f}, {2});
  Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});
  std::vector<tensorflow::Tensor> input_tensors;
  input_tensors.push_back(input_tensor);
  tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run(run_options, "serving_default", input_tensors,
                                &outputs));

  ASSERT_EQ(1, outputs.size());
  const auto& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

// Tests TfrtSavedModelFactory with native SavedModel with different
// configurations.
TEST_F(TfrtSavedModelFactoryTest, BasicWithSavedModelConfig) {
  std::unique_ptr<tfrt::SavedModel> saved_model;
  TfrtSavedModelConfig config;
  *config.add_saved_model_tags() = kSavedModelTagServe;
  model_path_ = test_util::TestSrcDirPath(
      "servables/tensorflow/"
      "testdata/saved_model_half_plus_two_cpu_with_saved_model_config/"
      "00000123");
  config.set_enable_saved_model_config(true);

  TF_ASSERT_OK(CreateTfrtSavedModel(config, &saved_model));

  Tensor input_tensor = test::AsTensor<float>({100.0f, 42.0f}, {2});
  Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});
  std::vector<tensorflow::Tensor> input_tensors;
  input_tensors.push_back(input_tensor);
  tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run(run_options, "serving_default", input_tensors,
                                &outputs));

  ASSERT_EQ(1, outputs.size());
  const auto& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

// Tests TfrtSavedModelFactory with native SavedModel with different
// configurations.
TEST_F(TfrtSavedModelFactoryTest, BasicWithSavedModelConfigAndGraphRewrite) {
  TF_ASSERT_OK(tensorflow::serving::ResetGraphRewriterForTesting());
  bool rewriter_was_called = false;
  TF_ASSERT_OK(tensorflow::serving::SetGraphRewriter([&](MetaGraphDef* graph) {
    rewriter_was_called = true;
    return absl::OkStatus();
  }));
  std::unique_ptr<tfrt::SavedModel> saved_model;
  TfrtSavedModelConfig config;
  *config.add_saved_model_tags() = kSavedModelTagServe;
  model_path_ = test_util::TestSrcDirPath(
      "servables/tensorflow/"
      "testdata/saved_model_half_plus_two_cpu_with_saved_model_config/"
      "00000123");
  config.set_enable_saved_model_config(true);

  TF_ASSERT_OK(CreateTfrtSavedModel(config, &saved_model));
  EXPECT_TRUE(rewriter_was_called);
  TF_ASSERT_OK(tensorflow::serving::ResetGraphRewriterForTesting());

  Tensor input_tensor = test::AsTensor<float>({100.0f, 42.0f}, {2});
  Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});
  std::vector<tensorflow::Tensor> input_tensors;
  input_tensors.push_back(input_tensor);
  tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run(run_options, "serving_default", input_tensors,
                                &outputs));

  ASSERT_EQ(1, outputs.size());
  const auto& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

TEST_F(TfrtSavedModelFactoryTest, Batch) {
  std::unique_ptr<tfrt::SavedModel> saved_model;
  TfrtSavedModelConfig config;
  config.mutable_batching_parameters()->mutable_max_batch_size()->set_value(4);
  config.mutable_batching_parameters()
      ->mutable_max_enqueued_batches()
      ->set_value(INT_MAX);
  config.mutable_batching_parameters()
      ->mutable_batch_timeout_micros()
      ->set_value(1000 * 1000 * 1000);
  config.mutable_batching_parameters()->mutable_num_batch_threads()->set_value(
      1);
  TF_ASSERT_OK(CreateTfrtSavedModel(config, &saved_model));

  Tensor input_tensor = test::AsTensor<float>({100.0f, 42.0f}, {2});
  Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});
  std::vector<tensorflow::Tensor> input_tensors;
  input_tensors.push_back(input_tensor);
  std::vector<tensorflow::Tensor> output_tensors1, output_tensors2;
  tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.

  {
    std::vector<std::unique_ptr<Thread>> request_threads;
    request_threads.reserve(2);
    request_threads.push_back(
        std::unique_ptr<Thread>(Env::Default()->StartThread(
            ThreadOptions(), strings::StrCat("thread_", 0),
            [&saved_model, &run_options, &input_tensors, &output_tensors1]() {
              TF_ASSERT_OK(saved_model->Run(run_options, "serving_default",
                                            input_tensors, &output_tensors1));
            })));
    request_threads.push_back(
        std::unique_ptr<Thread>(Env::Default()->StartThread(
            ThreadOptions(), strings::StrCat("thread_", 1),
            [&saved_model, &run_options, &input_tensors, &output_tensors2]() {
              TF_ASSERT_OK(saved_model->Run(run_options, "serving_default",
                                            input_tensors, &output_tensors2));
            })));
  }

  ASSERT_EQ(1, output_tensors1.size());
  test::ExpectTensorEqual<float>(expected_output, output_tensors1.at(0));

  ASSERT_EQ(1, output_tensors2.size());
  test::ExpectTensorEqual<float>(expected_output, output_tensors2.at(0));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
