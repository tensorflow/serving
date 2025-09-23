/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
#include "tensorflow_serving/servables/tensorflow/servable.h"

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/tensorflow/tfrt_regressor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
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

class TfrtRegressorTest : public ::testing::Test {
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

    request_ = test_util::CreateProto<RegressionRequest>(
        "model_spec {"
        "  name: \"test_model\""
        "  signature_name: \"regress_x_to_y\""
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

  absl::Status CallRegress(ServerCore* server_core,
                           const RegressionRequest& request,
                           RegressionResponse* response) {
    ServableHandle<Servable> servable;
    TF_RETURN_IF_ERROR(GetSavedModelServableHandle(server_core, &servable));
    tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.
    return RunRegress(
        run_options, kTestModelVersion,
        &(down_cast<TfrtSavedModelServable*>(servable.get()))->saved_model(),
        request, response);
  }

  static std::unique_ptr<ServerCore> server_core_;

  RegressionRequest request_;
};

std::unique_ptr<ServerCore> TfrtRegressorTest::server_core_;

TEST_F(TfrtRegressorTest, Basic) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"regress_x_to_y\""
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
  RegressionResponse response;

  TF_EXPECT_OK(CallRegress(server_core_.get(), request, &response));
  EXPECT_THAT(
      response,
      test_util::EqualsProto(
          "result { regressions { value: 42 } regressions { value: 12 }}"
          "model_spec {"
          "  name: \"test_model\""
          "  signature_name: \"regress_x_to_y\""
          "  version { value: 123 }"
          "}"));
}

TEST_F(TfrtRegressorTest, BasicWithContext) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"regress_x_to_y\""
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
  RegressionResponse response;

  TF_EXPECT_OK(CallRegress(server_core_.get(), request, &response));
  EXPECT_THAT(
      response,
      test_util::EqualsProto(
          "result { regressions { value: 42 } regressions { value: 12 }}"
          "model_spec {"
          "  name: \"test_model\""
          "  signature_name: \"regress_x_to_y\""
          "  version { value: 123 }"
          "}"));
}

TEST_F(TfrtRegressorTest, EmptyExampleList) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"regress_x_to_y\""
      "}"
      "input {"
      "  example_list {"
      "  }"
      "}");
  RegressionResponse response;

  absl::Status status = CallRegress(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtRegressorTest, EmptyExampleListWithContext) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"regress_x_to_y\""
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
  RegressionResponse response;

  absl::Status status = CallRegress(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtRegressorTest, EmptyInput) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"regress_x_to_y\""
      "}"
      "input {"
      "}");
  RegressionResponse response;

  absl::Status status = CallRegress(server_core_.get(), request, &response);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Input is empty"));
}

TEST_F(TfrtRegressorTest, InvalidFunctionName) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(std::nullopt));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), HasSubstr("not found"));
}

TEST_F(TfrtRegressorTest, InvalidFunctionUnmatchedInputSize) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs, "wrong input"};
  signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Expected one input Tensor."));
}

TEST_F(TfrtRegressorTest, InvalidFunctionUnmatchedOutputSize) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs, "wrong output"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Expected one output Tensor."));
}

TEST_F(TfrtRegressorTest, InvalidFunctionInvalidInputName) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"wrong input"};
  signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("No regression inputs found in function's metadata"));
}

TEST_F(TfrtRegressorTest, InvalidFunctionInvalidOutputName) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {"wrong output"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("No regression outputs found in function's metadata"));
}

TEST_F(TfrtRegressorTest, RunsFails) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(Return(errors::InvalidArgument("test error")));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("test error"));
}

TEST_F(TfrtRegressorTest, UnexpectedOutputTensorNumber) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs};
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
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected output_tensors and output_tensor_names to "
                        "have the same size."));
}

TEST_F(TfrtRegressorTest, UnexpectedOutputTensorShape) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs};
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
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected output Tensor shape to be either"));
}

TEST_F(TfrtRegressorTest, UnexpectedOutputTensorSize) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output(DT_FLOAT, TensorShape({3}));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Input batch size did not match output batch size"));
}

TEST_F(TfrtRegressorTest, UnexpectedOutputTensorType) {
  RegressionResponse response;
  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {kRegressInputs};
  signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output(DT_STRING, TensorShape({1}));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(output);
                }),
                Return(absl::OkStatus())));
  auto status = RunRegress(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                           saved_model.get(), request_, &response);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected output Tensor of DT_FLOAT."));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
