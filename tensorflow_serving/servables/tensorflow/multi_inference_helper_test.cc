/* Copyright 2018 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/multi_inference_helper.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

// Test fixture for MultiInferenceTest related tests sets up a ServerCore
// pointing to TF1 or TF2 version of half_plus_two SavedModel (based on `T`).
typedef std::integral_constant<int, 1> tf1_model_t;
typedef std::integral_constant<int, 2> tf2_model_t;

template <typename T>
class MultiInferenceTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    SetSignatureMethodNameCheckFeature(UseTf1Model());
    TF_ASSERT_OK(CreateServerCore(&server_core_));
  }

  static void TearDownTestSuite() { server_core_.reset(); }

 protected:
  static absl::Status CreateServerCore(
      std::unique_ptr<ServerCore>* server_core) {
    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    const auto& tf1_saved_model = test_util::TensorflowTestSrcDirPath(
        "cc/saved_model/testdata/half_plus_two");
    const auto& tf2_saved_model = test_util::TestSrcDirPath(
        "/servables/tensorflow/testdata/saved_model_half_plus_two_tf2_cpu");
    model_config->set_base_path(UseTf1Model() ? tf1_saved_model
                                              : tf2_saved_model);
    model_config->set_model_platform(kTensorFlowModelPlatform);

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;
    options.platform_config_map =
        CreateTensorFlowPlatformConfigMap(SessionBundleConfig());
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    return ServerCore::Create(std::move(options), server_core);
  }

  static bool UseTf1Model() { return std::is_same<T, tf1_model_t>::value; }

  ServerCore* GetServerCore() { return this->server_core_.get(); }

  const int64_t servable_version_ = kTestModelVersion;

 private:
  static std::unique_ptr<ServerCore> server_core_;
};

template <typename T>
std::unique_ptr<ServerCore> MultiInferenceTest<T>::server_core_;

TYPED_TEST_SUITE_P(MultiInferenceTest);

////////////////////////////////////////////////////////////////////////////////
// Test Helpers

void AddInput(const std::vector<std::pair<string, float>>& feature_kv,
              MultiInferenceRequest* request) {
  auto* example =
      request->mutable_input()->mutable_example_list()->add_examples();
  auto* features = example->mutable_features()->mutable_feature();
  for (const auto& feature : feature_kv) {
    (*features)[feature.first].mutable_float_list()->add_value(feature.second);
  }
}

void PopulateTask(const string& signature_name, const string& method_name,
                  int64_t version, InferenceTask* task) {
  ModelSpec model_spec;
  model_spec.set_name(kTestModelName);
  if (version > 0) {
    model_spec.mutable_version()->set_value(version);
  }
  model_spec.set_signature_name(signature_name);
  *task->mutable_model_spec() = model_spec;
  task->set_method_name(method_name);
}

void ExpectStatusError(const absl::Status& status,
                       const absl::StatusCode expected_code,
                       const string& message_substring) {
  ASSERT_EQ(expected_code, status.code());
  EXPECT_THAT(status.message(), ::testing::HasSubstr(message_substring));
}

////////////////////////////////////////////////////////////////////////////////
// Tests

TYPED_TEST_P(MultiInferenceTest, MissingInputTest) {
  MultiInferenceRequest request;
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;
  ExpectStatusError(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread::ThreadPoolOptions(), request,
                                      &response),
      static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
      "Input is empty");
}

TYPED_TEST_P(MultiInferenceTest, UndefinedSignatureTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("ThisSignatureDoesNotExist", kRegressMethodName, -1,
               request.add_tasks());

  MultiInferenceResponse response;
  ExpectStatusError(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread::ThreadPoolOptions(), request,
                                      &response),
      static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
      "signature not found");
}

// Two ModelSpecs, accessing different models.
TYPED_TEST_P(MultiInferenceTest, InconsistentModelSpecsInRequestTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  // Valid signature.
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());

  // Add invalid Task to request.
  ModelSpec model_spec;
  model_spec.set_name("ModelDoesNotExist");
  model_spec.set_signature_name("regress_x_to_y");
  auto* task = request.add_tasks();
  *task->mutable_model_spec() = model_spec;
  task->set_method_name(kRegressMethodName);

  MultiInferenceResponse response;
  ExpectStatusError(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread::ThreadPoolOptions(), request,
                                      &response),
      static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
      "must access the same model name");
}

TYPED_TEST_P(MultiInferenceTest, EvaluateDuplicateSignaturesTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());
  // Add the same task again (error).
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;
  ExpectStatusError(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread::ThreadPoolOptions(), request,
                                      &response),
      static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
      "Duplicate evaluation of signature: regress_x_to_y");
}

TYPED_TEST_P(MultiInferenceTest, UsupportedSignatureTypeTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("serving_default", kPredictMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;
  ExpectStatusError(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread::ThreadPoolOptions(), request,
                                      &response),
      static_cast<absl::StatusCode>(absl::StatusCode::kUnimplemented),
      "Unsupported signature");
}

TYPED_TEST_P(MultiInferenceTest, ValidSingleSignatureTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, this->servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;
  auto* inference_result = expected_response.add_results();
  auto* model_spec = inference_result->mutable_model_spec();
  *model_spec = request.tasks(0).model_spec();
  model_spec->mutable_version()->set_value(this->servable_version_);
  auto* regression_result = inference_result->mutable_regression_result();
  regression_result->add_regressions()->set_value(3.0);

  MultiInferenceResponse response;
  TF_ASSERT_OK(RunMultiInferenceWithServerCore(
      RunOptions(), this->GetServerCore(), thread::ThreadPoolOptions(), request,
      &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TYPED_TEST_P(MultiInferenceTest, MultipleValidRegressSignaturesTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, this->servable_version_,
               request.add_tasks());
  PopulateTask("regress_x_to_y2", kRegressMethodName, this->servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;

  // regress_x_to_y is y = 0.5x + 2.
  auto* inference_result_1 = expected_response.add_results();
  auto* model_spec_1 = inference_result_1->mutable_model_spec();
  *model_spec_1 = request.tasks(0).model_spec();
  model_spec_1->mutable_version()->set_value(this->servable_version_);
  auto* regression_result_1 = inference_result_1->mutable_regression_result();
  regression_result_1->add_regressions()->set_value(3.0);

  // regress_x_to_y2 is y2 = 0.5x + 3.
  auto* inference_result_2 = expected_response.add_results();
  auto* model_spec_2 = inference_result_2->mutable_model_spec();
  *model_spec_2 = request.tasks(1).model_spec();
  model_spec_2->mutable_version()->set_value(this->servable_version_);
  auto* regression_result_2 = inference_result_2->mutable_regression_result();
  regression_result_2->add_regressions()->set_value(4.0);

  MultiInferenceResponse response;
  TF_ASSERT_OK(RunMultiInferenceWithServerCore(
      RunOptions(), this->GetServerCore(), thread::ThreadPoolOptions(), request,
      &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TYPED_TEST_P(MultiInferenceTest, RegressAndClassifySignaturesTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, this->servable_version_,
               request.add_tasks());
  PopulateTask("classify_x_to_y", kClassifyMethodName, this->servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;
  auto* inference_result_1 = expected_response.add_results();
  auto* model_spec_1 = inference_result_1->mutable_model_spec();
  *model_spec_1 = request.tasks(0).model_spec();
  model_spec_1->mutable_version()->set_value(this->servable_version_);
  auto* regression_result = inference_result_1->mutable_regression_result();
  regression_result->add_regressions()->set_value(3.0);

  auto* inference_result_2 = expected_response.add_results();
  auto* model_spec_2 = inference_result_2->mutable_model_spec();
  *model_spec_2 = request.tasks(1).model_spec();
  model_spec_2->mutable_version()->set_value(this->servable_version_);
  auto* classification_result =
      inference_result_2->mutable_classification_result();
  classification_result->add_classifications()->add_classes()->set_score(3.0);

  MultiInferenceResponse response;
  TF_ASSERT_OK(RunMultiInferenceWithServerCore(
      RunOptions(), this->GetServerCore(), thread::ThreadPoolOptions(), request,
      &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Verifies that RunMultiInferenceWithServerCoreWithModelSpec() uses the model
// spec override rather than the one in the request.
TYPED_TEST_P(MultiInferenceTest, ModelSpecOverride) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, this->servable_version_,
               request.add_tasks());
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  MultiInferenceResponse response;
  EXPECT_NE(tensorflow::error::NOT_FOUND,
            RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                            thread::ThreadPoolOptions(),
                                            request, &response)
                .code());
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      RunMultiInferenceWithServerCoreWithModelSpec(
          RunOptions(), this->GetServerCore(), thread::ThreadPoolOptions(),
          model_spec_override, request, &response)
          .code());
}

TYPED_TEST_P(MultiInferenceTest, ThreadPoolOptions) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, this->servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;
  auto* inference_result = expected_response.add_results();
  auto* model_spec = inference_result->mutable_model_spec();
  *model_spec = request.tasks(0).model_spec();
  model_spec->mutable_version()->set_value(this->servable_version_);
  auto* regression_result = inference_result->mutable_regression_result();
  regression_result->add_regressions()->set_value(3.0);

  test_util::CountingThreadPool inter_op_threadpool(Env::Default(), "InterOp",
                                                    /*num_threads=*/1);
  test_util::CountingThreadPool intra_op_threadpool(Env::Default(), "IntraOp",
                                                    /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options;
  thread_pool_options.inter_op_threadpool = &inter_op_threadpool;
  thread_pool_options.intra_op_threadpool = &intra_op_threadpool;
  MultiInferenceResponse response;
  TF_ASSERT_OK(
      RunMultiInferenceWithServerCore(RunOptions(), this->GetServerCore(),
                                      thread_pool_options, request, &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));

  // The intra_op_threadpool doesn't have anything scheduled.
  ASSERT_GE(inter_op_threadpool.NumScheduled(), 1);
}

REGISTER_TYPED_TEST_SUITE_P(
    MultiInferenceTest, MissingInputTest, UndefinedSignatureTest,
    InconsistentModelSpecsInRequestTest, EvaluateDuplicateSignaturesTest,
    UsupportedSignatureTypeTest, ValidSingleSignatureTest,
    MultipleValidRegressSignaturesTest, RegressAndClassifySignaturesTest,
    ModelSpecOverride, ThreadPoolOptions);

typedef ::testing::Types<tf1_model_t, tf2_model_t> ModelTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(MultiInference, MultiInferenceTest, ModelTypes);

}  // namespace
}  // namespace serving
}  // namespace tensorflow
