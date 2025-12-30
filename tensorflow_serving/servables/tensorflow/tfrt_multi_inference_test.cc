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

#include "tensorflow_serving/servables/tensorflow/tfrt_multi_inference.h"

#include "absl/status/status.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

class TfrtMultiInferenceTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));
    CreateServerCore(&server_core_);
  }

  static void TearDownTestSuite() { server_core_.reset(); }

 protected:
  static void CreateServerCore(std::unique_ptr<ServerCore>* server_core) {
    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(
        test_util::TestSrcDirPath("servables/tensorflow/"
                                  "testdata/saved_model_half_plus_two_cpu"));
    model_config->set_model_platform(kTensorFlowModelPlatform);

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;
    PlatformConfigMap platform_config_map;
    ::google::protobuf::Any source_adapter_config;
    TfrtSavedModelSourceAdapterConfig saved_model_bundle_source_adapter_config;
    source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
    (*(*platform_config_map
            .mutable_platform_configs())[kTensorFlowModelPlatform]
          .mutable_source_adapter_config()) = source_adapter_config;
    options.platform_config_map = platform_config_map;
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core_));
  }

  ServerCore* GetServerCore() { return server_core_.get(); }

  absl::Status GetServableHandle(ServableHandle<Servable>* servable) {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    return GetServerCore()->GetServableHandle(model_spec, servable);
  }

  const int64_t servable_version_ = kTestModelVersion;

 private:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> TfrtMultiInferenceTest::server_core_;

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

TEST_F(TfrtMultiInferenceTest, MissingInputTest) {
  MultiInferenceRequest request;
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;

  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  ExpectStatusError(
      RunMultiInference(
          tfrt::SavedModel::RunOptions(), servable_version_,
          &(down_cast<TfrtSavedModelServable*>(servable.get()))  // NOLINT
               ->saved_model(),
          request, &response),
      absl::StatusCode::kInvalidArgument, "Input is empty");
}

TEST_F(TfrtMultiInferenceTest, UndefinedSignatureTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("ThisSignatureDoesNotExist", kRegressMethodName, -1,
               request.add_tasks());

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  ExpectStatusError(servable->MultiInference({}, request, &response),
                    absl::StatusCode::kInvalidArgument, "not found");
}

// Two ModelSpecs, accessing different models.
TEST_F(TfrtMultiInferenceTest, InconsistentModelSpecsInRequestTest) {
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
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  ExpectStatusError(servable->MultiInference({}, request, &response),
                    absl::StatusCode::kInvalidArgument,
                    "must access the same model name");
}

TEST_F(TfrtMultiInferenceTest, EvaluateDuplicateFunctionsTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());
  // Add the same task again (error).
  PopulateTask("regress_x_to_y", kRegressMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  ExpectStatusError(servable->MultiInference({}, request, &response),
                    absl::StatusCode::kInvalidArgument,
                    "Duplicate evaluation of signature: regress_x_to_y");
}

TEST_F(TfrtMultiInferenceTest, UsupportedSignatureTypeTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("serving_default", kPredictMethodName, -1, request.add_tasks());

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  ExpectStatusError(servable->MultiInference({}, request, &response),
                    absl::StatusCode::kUnimplemented, "Unsupported signature");
}

TEST_F(TfrtMultiInferenceTest, ValidSingleSignatureTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;
  auto* inference_result = expected_response.add_results();
  auto* model_spec = inference_result->mutable_model_spec();
  *model_spec = request.tasks(0).model_spec();
  model_spec->mutable_version()->set_value(servable_version_);
  auto* regression_result = inference_result->mutable_regression_result();
  regression_result->add_regressions()->set_value(3.0);

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  TF_ASSERT_OK(servable->MultiInference({}, request, &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TEST_F(TfrtMultiInferenceTest, MultipleValidRegressSignaturesTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, servable_version_,
               request.add_tasks());
  PopulateTask("regress_x_to_y2", kRegressMethodName, servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;

  // regress_x_to_y is y = 0.5x + 2.
  auto* inference_result_1 = expected_response.add_results();
  auto* model_spec_1 = inference_result_1->mutable_model_spec();
  *model_spec_1 = request.tasks(0).model_spec();
  model_spec_1->mutable_version()->set_value(servable_version_);
  auto* regression_result_1 = inference_result_1->mutable_regression_result();
  regression_result_1->add_regressions()->set_value(3.0);

  // regress_x_to_y2 is y2 = 0.5x + 3.
  auto* inference_result_2 = expected_response.add_results();
  auto* model_spec_2 = inference_result_2->mutable_model_spec();
  *model_spec_2 = request.tasks(1).model_spec();
  model_spec_2->mutable_version()->set_value(servable_version_);
  auto* regression_result_2 = inference_result_2->mutable_regression_result();
  regression_result_2->add_regressions()->set_value(4.0);

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  TF_ASSERT_OK(servable->MultiInference({}, request, &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TEST_F(TfrtMultiInferenceTest, RegressAndClassifySignaturesTest) {
  MultiInferenceRequest request;
  AddInput({{"x", 2}}, &request);
  PopulateTask("regress_x_to_y", kRegressMethodName, servable_version_,
               request.add_tasks());
  PopulateTask("classify_x_to_y", kClassifyMethodName, servable_version_,
               request.add_tasks());

  MultiInferenceResponse expected_response;
  auto* inference_result_1 = expected_response.add_results();
  auto* model_spec_1 = inference_result_1->mutable_model_spec();
  *model_spec_1 = request.tasks(0).model_spec();
  model_spec_1->mutable_version()->set_value(servable_version_);
  auto* regression_result = inference_result_1->mutable_regression_result();
  regression_result->add_regressions()->set_value(3.0);

  auto* inference_result_2 = expected_response.add_results();
  auto* model_spec_2 = inference_result_2->mutable_model_spec();
  *model_spec_2 = request.tasks(1).model_spec();
  model_spec_2->mutable_version()->set_value(servable_version_);
  auto* classification_result =
      inference_result_2->mutable_classification_result();
  classification_result->add_classifications()->add_classes()->set_score(3.0);

  MultiInferenceResponse response;
  ServableHandle<Servable> servable;
  TF_ASSERT_OK(GetServableHandle(&servable));
  TF_ASSERT_OK(servable->MultiInference({}, request, &response));
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
