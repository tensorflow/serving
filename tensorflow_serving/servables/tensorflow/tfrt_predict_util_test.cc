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

#include "tensorflow_serving/servables/tensorflow/tfrt_predict_util.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/test_util/mock_tfrt_saved_model.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/oss_or_google.h"

namespace tensorflow {
namespace serving {
namespace {
using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::WithArgs;

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

const char kInputTensorKey[] = "x";
const char kOutputTensorKey[] = "y";

class PredictImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));

    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(
        test_util::TestSrcDirPath("servables/tensorflow/testdata/"
                                  "saved_model_half_plus_two_tf2_cpu"));
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
    TF_ASSERT_OK(
        ServerCore::Create(std::move(options), &saved_model_server_core_));
  }

  static void TearDownTestSuite() { saved_model_server_core_.reset(); }

 protected:
  absl::Status GetSavedModelServableHandle(ServerCore* server_core,
                                           ServableHandle<Servable>* servable) {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    return server_core->GetServableHandle(model_spec, servable);
  }

  ServerCore* GetServerCore() { return saved_model_server_core_.get(); }

  absl::Status CallPredict(ServerCore* server_core,
                           const PredictRequest& request,
                           PredictResponse* response,
                           absl::Duration timeout = absl::ZeroDuration()) {
    ServableHandle<Servable> servable;
    TF_RETURN_IF_ERROR(GetSavedModelServableHandle(server_core, &servable));

    // Set deadline in run options.
    Servable::RunOptions run_options;
    if (timeout != absl::ZeroDuration())
      run_options.deadline = absl::Now() + timeout;
    return servable->Predict(run_options, request, response);
  }

 private:
  static std::unique_ptr<ServerCore> saved_model_server_core_;
};

std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_;

TEST_F(PredictImplTest, PredictionSuccess) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  TF_EXPECT_OK(CallPredict(GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(3);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  *expected_response.mutable_model_spec() = *model_spec;
  expected_response.mutable_model_spec()->set_signature_name(
      kDefaultServingSignatureDefKey);
  (*expected_response.mutable_outputs())[kOutputTensorKey] =
      output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TEST_F(PredictImplTest, PredictionSuccessWithDefaultInputs) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // prediction result = 0.5x + 2 = 2 with x defaults to 0.
  TF_EXPECT_OK(CallPredict(GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(2);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  PredictResponse expected_response;
  *expected_response.mutable_model_spec() = *model_spec;
  expected_response.mutable_model_spec()->set_signature_name(
      kDefaultServingSignatureDefKey);
  (*expected_response.mutable_outputs())[kOutputTensorKey] =
      output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

TEST_F(PredictImplTest, PredictionInvalidTensor) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_bool_val(true);
  tensor_proto.set_dtype(tensorflow::DT_BOOL);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  auto status = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.message(), HasSubstr("Expected input x to be float"));
}

TEST_F(PredictImplTest, PredictionMissingFunction) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(std::nullopt));
  auto status =
      RunPredict(tfrt_stub::SavedModel::RunOptions(), kTestModelVersion,
                 saved_model.get(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::FAILED_PRECONDITION);
  EXPECT_THAT(status.message(), HasSubstr("not found"));
}

TEST_F(PredictImplTest, PredictionMissingInput) {
  PredictRequest request;
  request.mutable_model_spec()->set_name(kTestModelName);
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"unknown"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  auto status =
      RunPredict(tfrt_stub::SavedModel::RunOptions(), kTestModelVersion,
                 saved_model.get(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Request inputs do not match required inputs for model "
          "`test_model`. Send extra: {x}. Missing but required: {unknown}."));
}

TEST_F(PredictImplTest, PredictionRunError) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  EXPECT_CALL(*saved_model,
              Run(_, _, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(1)
      .WillRepeatedly(Return(errors::InvalidArgument("test error")));
  auto status =
      RunPredict(tfrt_stub::SavedModel::RunOptions(), kTestModelVersion,
                 saved_model.get(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.message(), HasSubstr("test error"));
}

TEST_F(PredictImplTest, PredictionUnmatchedOutputNumber) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
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
  auto status =
      RunPredict(tfrt_stub::SavedModel::RunOptions(), kTestModelVersion,
                 saved_model.get(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::UNKNOWN);
  EXPECT_THAT(status.message(), HasSubstr("Predict internal error."));
}

TEST_F(PredictImplTest, OutputFilters) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter("output1");

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
  signature.output_names = {"output1", "output2"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));

  tensorflow::SignatureDef signature_def;
  tensorflow::TensorInfo tensor_info1, tensor_info2, tensor_info3;
  tensor_info1.set_name("x");
  tensor_info2.set_name("output1");
  tensor_info3.set_name("output2");
  signature_def.mutable_inputs()->insert({kInputTensorKey, tensor_info1});
  signature_def.mutable_outputs()->insert({"output1", tensor_info2});
  signature_def.mutable_outputs()->insert({"output2", tensor_info3});
  signature_def.set_method_name("tensorflow/serving/predict");

  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.mutable_signature_def()->insert(
      {"serving_default", signature_def});
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .Times(1)
      .WillRepeatedly(ReturnRef(meta_graph_def));

  TensorProto output_tensor_proto1;
  output_tensor_proto1.add_float_val(1.0);
  output_tensor_proto1.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto1.mutable_tensor_shape();

  EXPECT_CALL(
      *saved_model,
      RunByTensorNames(_, _, ::testing::SizeIs(1),
                       ::testing::An<absl::Span<const std::string>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<4>([&](std::vector<Tensor>* output_tensors) {
                  Tensor output_tensor;
                  CHECK(output_tensor.FromProto(output_tensor_proto1));
                  output_tensors->push_back(output_tensor);
                }),
                Return(absl::OkStatus())));
  TF_EXPECT_OK(RunPredict(tfrt_stub::SavedModel::RunOptions(),
                          kTestModelVersion, saved_model.get(), request,
                          &response));
  EXPECT_EQ(response.outputs_size(), 1);
  EXPECT_TRUE(response.outputs().find("output1") != response.outputs().end());
  EXPECT_THAT(response.outputs().at("output1"),
              test_util::EqualsProto(output_tensor_proto1));
}

TEST_F(PredictImplTest, OutputFiltersFullSet) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter("output1");
  request.add_output_filter("output2");

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
  signature.output_names = {"output1", "output2"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));

  // if the output_filter is a full set, we should still call Run(), since full
  // set is equivalent to an empty filter.
  EXPECT_CALL(*saved_model, GetMetaGraphDef()).Times(0);
  EXPECT_CALL(
      *saved_model,
      RunByTensorNames(_, _, ::testing::SizeIs(1),
                       ::testing::An<absl::Span<const std::string>>(), _))
      .Times(0);
  TensorProto output_tensor_proto1;
  output_tensor_proto1.add_float_val(1.0);
  output_tensor_proto1.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto1.mutable_tensor_shape();
  TensorProto output_tensor_proto2;
  output_tensor_proto2.add_float_val(2.0);
  output_tensor_proto2.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto2.mutable_tensor_shape();
  EXPECT_CALL(*saved_model, Run(_, _, _, _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  Tensor output_tensor1;
                  CHECK(output_tensor1.FromProto(output_tensor_proto1));
                  output_tensors->push_back(output_tensor1);
                  Tensor output_tensor2;
                  CHECK(output_tensor2.FromProto(output_tensor_proto2));
                  output_tensors->push_back(output_tensor2);
                }),
                Return(absl::OkStatus())));

  TF_EXPECT_OK(RunPredict(tfrt_stub::SavedModel::RunOptions(),
                          kTestModelVersion, saved_model.get(), request,
                          &response));
  EXPECT_EQ(response.outputs_size(), 2);
  EXPECT_TRUE(response.outputs().find("output1") != response.outputs().end());
  EXPECT_THAT(response.outputs().at("output1"),
              test_util::EqualsProto(output_tensor_proto1));
  EXPECT_TRUE(response.outputs().find("output2") != response.outputs().end());
  EXPECT_THAT(response.outputs().at("output2"),
              test_util::EqualsProto(output_tensor_proto2));
}

TEST_F(PredictImplTest, OutputFiltersWithDefaultInputs) {
  PredictRequest request;
  PredictResponse response;

  request.add_output_filter("output1");

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
  signature.output_names = {"output1", "output2"};
  Tensor tensor(0);
  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  signature.default_inputs[kInputTensorKey] = tensor_proto;
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));

  tensorflow::SignatureDef signature_def;
  tensorflow::TensorInfo tensor_info1, tensor_info2, tensor_info3;
  tensor_info1.set_name("x");
  tensor_info2.set_name("output1");
  tensor_info3.set_name("output2");
  signature_def.mutable_inputs()->insert({kInputTensorKey, tensor_info1});
  signature_def.mutable_outputs()->insert({"output1", tensor_info2});
  signature_def.mutable_outputs()->insert({"output2", tensor_info3});
  signature_def.set_method_name("tensorflow/serving/predict");
  (*signature_def.mutable_defaults())[kInputTensorKey] = tensor_proto;

  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.mutable_signature_def()->insert(
      {"serving_default", signature_def});
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .Times(1)
      .WillRepeatedly(ReturnRef(meta_graph_def));

  TensorProto output_tensor_proto1;
  output_tensor_proto1.add_float_val(1.0);
  output_tensor_proto1.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto1.mutable_tensor_shape();

  EXPECT_CALL(
      *saved_model,
      RunByTensorNames(_, ::testing::SizeIs(1), ::testing::SizeIs(1),
                       ::testing::An<absl::Span<const std::string>>(), _))
      .Times(1)
      .WillRepeatedly(
          DoAll(WithArgs<4>([&](std::vector<Tensor>* output_tensors) {
                  Tensor output_tensor;
                  CHECK(output_tensor.FromProto(output_tensor_proto1));
                  output_tensors->push_back(output_tensor);
                }),
                Return(absl::OkStatus())));
  TF_EXPECT_OK(RunPredict(tfrt::SavedModel::RunOptions(), kTestModelVersion,
                          saved_model.get(), request, &response));
  EXPECT_EQ(response.outputs_size(), 1);
  EXPECT_TRUE(response.outputs().find("output1") != response.outputs().end());
  EXPECT_THAT(response.outputs().at("output1"),
              test_util::EqualsProto(output_tensor_proto1));
}

TEST_F(PredictImplTest, UnmatchedOutputFilters) {
  PredictRequest request;
  PredictResponse response;

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter("output1");
  request.add_output_filter("output3");

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature signature;
  signature.input_names = {"x"};
  tfrt::TensorSpec spec(tensorflow::DT_FLOAT);
  signature.input_specs = {spec};
  signature.output_names = {"output1", "output2"};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(_))
      .Times(1)
      .WillRepeatedly(Return(function_metadata));
  Tensor output_tensor;

  tensorflow::SignatureDef signature_def;
  tensorflow::TensorInfo tensor_info1, tensor_info2, tensor_info3;
  tensor_info1.set_name("x");
  tensor_info2.set_name("output1");
  tensor_info3.set_name("output2");
  signature_def.mutable_inputs()->insert({kInputTensorKey, tensor_info1});
  signature_def.mutable_outputs()->insert({"output1", tensor_info2});
  signature_def.mutable_outputs()->insert({"output2", tensor_info3});
  signature_def.set_method_name("tensorflow/serving/predict");

  tensorflow::MetaGraphDef meta_graph_def;
  meta_graph_def.mutable_signature_def()->insert(
      {"serving_default", signature_def});
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .Times(1)
      .WillRepeatedly(ReturnRef(meta_graph_def));

  auto status =
      RunPredict(tfrt_stub::SavedModel::RunOptions(), kTestModelVersion,
                 saved_model.get(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.message(),
      HasSubstr("output tensor alias not found in signature: output3 Outputs "
                "expected to be in the set {output1,output2}."));
}

TEST_F(PredictImplTest, PredictionTimeout) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  // Set the deadline to be 1 nanosecond from now. This makes the timer
  // timeout before the request completes.
  auto status =
      CallPredict(GetServerCore(), request, &response, absl::Nanoseconds(1));

  EXPECT_EQ(status.code(), tensorflow::error::Code::DEADLINE_EXCEEDED);
  EXPECT_EQ(status.message(), "Deadline exceeded.");
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
