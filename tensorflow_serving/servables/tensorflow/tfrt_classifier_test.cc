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

#include "tensorflow_serving/servables/tensorflow/tfrt_classifier.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/env.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/test_util/mock_tfrt_saved_model.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::WithArgs;

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

class TfrtClassifierTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));
  }

  void SetUp() override {
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

    request_ = test_util::CreateProto<ClassificationRequest>(
        "model_spec {"
        "  name: \"test_model\""
        "  signature_name: \"classify_x_to_y\""
        "}"
        "input {"
        "  example_list {"
        "    examples {"
        "      features {"
        "        feature: {"
        "          key  : \"x\""
        "          value: {"
        "            float_list: {"
        "              value: [ 20.0 ]"
        "            }"
        "          }"
        "        }"
        "      }"
        "    }"
        "  }"
        "}");
  }

  static void TearDownTestSuite() { server_core_ = nullptr; }

 protected:
  absl::Status GetSavedModelServableHandle(ServerCore* server_core,
                                           ServableHandle<Servable>* servable) {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    return server_core->GetServableHandle(model_spec, servable);
  }

  absl::Status CallClassify(ServerCore* server_core,
                            const ClassificationRequest& request,
                            ClassificationResponse* response) {
    ServableHandle<Servable> servable;
    TF_RETURN_IF_ERROR(GetSavedModelServableHandle(server_core, &servable));
    return servable->Classify({}, request, response);
  }

  // Classifier valid after calling create.
  std::unique_ptr<ClassifierInterface> classifier_;
  static std::unique_ptr<ServerCore> server_core_;
  ClassificationRequest request_;
};

std::unique_ptr<ServerCore> TfrtClassifierTest::server_core_;

TEST_F(TfrtClassifierTest, Basic) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"classify_x_to_y\""
      "}"
      "input {"
      "  example_list {"
      "    examples {"
      "      features {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 80.0 ]"
      "            }"
      "          }"
      "        }"
      "        feature: {"
      "          key  : \"locale\""
      "          value: {"
      "            bytes_list: {"
      "              value: [ \"pt_BR\" ]"
      "            }"
      "          }"
      "        }"
      "        feature: {"
      "          key  : \"age\""
      "          value: {"
      "            float_list: {"
      "              value: [ 19.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "    examples {"
      "      features {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 20.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "  }"
      "}");
  ClassificationResponse response;

  TF_EXPECT_OK(CallClassify(server_core_.get(), request, &response));
  EXPECT_THAT(response,
              test_util::EqualsProto(
                  "result { classifications { classes { "
                  "score: 42 } } classifications { classes {score: 12 } } }"
                  "model_spec {"
                  "  name: \"test_model\""
                  "  signature_name: \"classify_x_to_y\""
                  "  version { value: 123 }"
                  "}"));
}

TEST_F(TfrtClassifierTest, BasicWithContext) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"classify_x_to_y\""
      "}"
      "input {"
      "   example_list_with_context {"
      "    examples {"
      "      features {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 80.0 ]"
      "            }"
      "          }"
      "        }"
      "        feature: {"
      "          key  : \"locale\""
      "          value: {"
      "            bytes_list: {"
      "              value: [ \"pt_BR\" ]"
      "            }"
      "          }"
      "        }"
      "        feature: {"
      "          key  : \"age\""
      "          value: {"
      "            float_list: {"
      "              value: [ 19.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "    examples {"
      "      features {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 20.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "    context: {"
      "      features: {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 10.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "  }"
      "}");
  ClassificationResponse response;

  TF_EXPECT_OK(CallClassify(server_core_.get(), request, &response));
  EXPECT_THAT(response, test_util::EqualsProto(
                            "result { classifications { classes { score: 42 }} "
                            "classifications { classes { score: 12 }}}"
                            "model_spec {"
                            "  name: \"test_model\""
                            "  signature_name: \"classify_x_to_y\""
                            "  version { value: 123 }"
                            "}"));
}

TEST_F(TfrtClassifierTest, EmptyExampleList) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"classify_x_to_y\""
      "}"
      "input {"
      "  example_list {"
      "  }"
      "}");
  ClassificationResponse response;

  absl::Status status = CallClassify(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtClassifierTest, EmptyExampleListWithContext) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"classify_x_to_y\""
      "}"
      "input {"
      "  example_list_with_context {"
      "    context: {"
      "      features: {"
      "        feature: {"
      "          key  : \"x\""
      "          value: {"
      "            float_list: {"
      "              value: [ 10.0 ]"
      "            }"
      "          }"
      "        }"
      "      }"
      "    }"
      "  }"
      "}");
  ClassificationResponse response;

  absl::Status status = CallClassify(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtClassifierTest, EmptyInput) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"classify_x_to_y\""
      "}"
      "input {"
      "}");
  ClassificationResponse response;

  absl::Status status = CallClassify(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtClassifierTest, InvalidFunctionName) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(std::nullopt));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), HasSubstr("not found"));
}

TEST_F(TfrtClassifierTest, InvalidFunctionUnmatchedInputSize) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs, "wrong input"};
  signature.output_names = {kClassifyOutputClasses, kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Expected one input Tensor."));
}

TEST_F(TfrtClassifierTest, InvalidFunctionUnmatchedOutputSize) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputClasses, kClassifyOutputScores,
                            "wrong output"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected one or two output Tensors"));
}

TEST_F(TfrtClassifierTest, InvalidFunctionInvalidInputName) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"wrong input"};
  signature.output_names = {kClassifyOutputClasses, kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.message(),
      HasSubstr("No classification inputs found in function's metadata"));
}

TEST_F(TfrtClassifierTest, InvalidFunctionInvalidOutputName) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {"wrong output", kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected classification function outputs to contain"));
}

TEST_F(TfrtClassifierTest, RunsFails) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputClasses, kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(Return(errors::InvalidArgument("test error")));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("test error"));
}

TEST_F(TfrtClassifierTest, UnexpectedOutputTensorNumber) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputClasses, kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output;
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Unexpected output tensors size"));
}

TEST_F(TfrtClassifierTest, UnexpectedOutputTensorShape) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output(DT_FLOAT, TensorShape({1, 1, 1}));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Expected Tensor shape"));
}

TEST_F(TfrtClassifierTest, UnexpectedOutputTensorType) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output(DT_STRING, TensorShape({1, 1}));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected scores Tensor of DT_FLOAT"));
}

TEST_F(TfrtClassifierTest, UnexpectedOutputTensorSize) {
  ClassificationResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kClassifyInputs};
  signature.output_names = {kClassifyOutputScores};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output(DT_FLOAT, TensorShape({10, 1}));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunClassify(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                            saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected scores output batch size of"));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
