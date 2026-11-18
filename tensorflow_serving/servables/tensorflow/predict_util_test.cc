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

#include "tensorflow_serving/servables/tensorflow/predict_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/oss_or_google.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

const char kInputTensorKey[] = "x";
const char kOutputTensorKey[] = "y";

// Fake Session, that copies input tensors to output.
class FakeSession : public tensorflow::Session {
 public:
  FakeSession() {}
  ~FakeSession() override = default;
  absl::Status Create(const GraphDef& graph) override {
    return absl::UnimplementedError("not available in fake");
  }
  absl::Status Extend(const GraphDef& graph) override {
    return absl::UnimplementedError("not available in fake");
  }
  absl::Status Close() override {
    return absl::UnimplementedError("not available in fake");
  }
  absl::Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return absl::UnimplementedError("not available in fake");
  }
  absl::Status Run(const std::vector<std::pair<std::string, Tensor>>& inputs,
                   const std::vector<std::string>& output_names,
                   const std::vector<std::string>& target_nodes,
                   std::vector<Tensor>* outputs) override {
    RunMetadata run_metadata;
    return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
               &run_metadata);
  }
  absl::Status Run(const RunOptions& run_options,
                   const std::vector<std::pair<std::string, Tensor>>& inputs,
                   const std::vector<std::string>& output_names,
                   const std::vector<std::string>& target_nodes,
                   std::vector<Tensor>* outputs,
                   RunMetadata* run_metadata) override {
    return Run(run_options, inputs, output_names, target_nodes, outputs,
               run_metadata, thread::ThreadPoolOptions());
  }
  absl::Status Run(
      const RunOptions& run_options,
      const std::vector<std::pair<std::string, Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      const std::vector<std::string>& target_nodes,
      std::vector<Tensor>* outputs, RunMetadata* run_metadata,
      const thread::ThreadPoolOptions& thread_pool_options) override {
    if (!inputs.empty()) {
      outputs->assign(output_names.size(), inputs[0].second);
    }
    return absl::OkStatus();
  }
};

class PredictImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    if (!IsTensorflowServingOSS()) {
      const std::string bad_half_plus_two_path = test_util::TestSrcDirPath(
          "/servables/tensorflow/testdata/bad_half_plus_two");
      TF_ASSERT_OK(CreateServerCore(bad_half_plus_two_path,
                                    &saved_model_server_core_bad_model_));
    }

    TF_ASSERT_OK(CreateServerCore(test_util::TensorflowTestSrcDirPath(
                                      "cc/saved_model/testdata/half_plus_two"),
                                  &saved_model_server_core_));
    TF_ASSERT_OK(CreateServerCore(
        test_util::TestSrcDirPath(
            "/servables/tensorflow/testdata/saved_model_counter"),
        &saved_model_server_core_counter_model_));
  }

  static void TearDownTestSuite() {
    saved_model_server_core_.reset();
    saved_model_server_core_bad_model_.reset();
    saved_model_server_core_counter_model_.reset();
  }

 protected:
  static absl::Status CreateServerCore(
      const std::string& model_path, std::unique_ptr<ServerCore>* server_core) {
    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(model_path);
    model_config->set_model_platform(kTensorFlowModelPlatform);

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;
    options.platform_config_map =
        CreateTensorFlowPlatformConfigMap(SessionBundleConfig());
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    return ServerCore::Create(std::move(options), server_core);
  }

  ServerCore* GetServerCore() {
    return saved_model_server_core_.get();
  }

  ServerCore* GetServerCoreWithBadModel() {
    return saved_model_server_core_bad_model_.get();
  }

  ServerCore* GetServerCoreWithCounterModel() {
    return saved_model_server_core_counter_model_.get();
  }

  absl::Status GetSavedModelServableHandle(
      ServerCore* server_core, ServableHandle<SavedModelBundle>* bundle) {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    return server_core->GetServableHandle(model_spec, bundle);
  }

  absl::Status CallPredict(
      ServerCore* server_core, const PredictRequest& request,
      PredictResponse* response,
      const thread::ThreadPoolOptions& thread_pool_options =
          thread::ThreadPoolOptions()) {
    ServableHandle<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(GetSavedModelServableHandle(server_core, &bundle));
    return RunPredict(GetRunOptions(), bundle->meta_graph_def,
                      kTestModelVersion, bundle->session.get(), request,
                      response, thread_pool_options);
  }

  RunOptions GetRunOptions() { return RunOptions(); }

 private:
  static std::unique_ptr<ServerCore> saved_model_server_core_;
  static std::unique_ptr<ServerCore> saved_model_server_core_bad_model_;
  static std::unique_ptr<ServerCore> saved_model_server_core_counter_model_;
};

std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_;
std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_bad_model_;
std::unique_ptr<ServerCore>
    PredictImplTest::saved_model_server_core_counter_model_;

TEST_F(PredictImplTest, MissingOrEmptyModelSpec) {
  PredictRequest request;
  PredictResponse response;

  // Empty request is invalid.
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            CallPredict(GetServerCore(), request, &response).code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            CallPredict(GetServerCore(), request, &response).code());

  // Model name is wrong.
  model_spec->set_name("test");
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            CallPredict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, EmptyInputList) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // The input is empty.
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            CallPredict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, InputTensorsDontMatchModelSpecInputs) {
  PredictRequest request;
  PredictResponse response;
  auto inputs = request.mutable_inputs();

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto1;
  tensor_proto1.add_string_val("any_value");
  tensor_proto1.set_dtype(tensorflow::DT_STRING);
  tensor_proto1.mutable_tensor_shape()->add_dim()->set_size(1);
  (*inputs)["unknown_key1"] = tensor_proto1;

  TensorProto tensor_proto2;
  tensor_proto2.add_float_val(1.0);
  tensor_proto2.set_dtype(tensorflow::DT_FLOAT);
  tensor_proto2.mutable_tensor_shape()->add_dim()->set_size(1);
  (*inputs)["unknown_key2"] = tensor_proto2;

  absl::Status status = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status.code(),
            static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Sent extra: {unknown_key1,unknown_key2}"));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr(absl::StrCat("Missing but required: {",
                                                kInputTensorKey, "}")));
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
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("Expects arg[0] to be float but bool is provided"));
}

TEST_F(PredictImplTest, OutputFiltersDontMatchModelSpecOutputs) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter("output_filter");

  // Output filter like this doesn't exist.
  absl::Status status1 = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status1.code(),
            static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status1.message(),
              ::testing::HasSubstr(
                  "output tensor alias not found in signature: output_filter"));

  request.clear_output_filter();
  request.add_output_filter(kOutputTensorKey);
  TF_EXPECT_OK(CallPredict(GetServerCore(), request, &response));
  request.add_output_filter(kOutputTensorKey);

  // Duplicate output filter specified.
  absl::Status status2 = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status2.code(),
            static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status2.message(),
              ::testing::HasSubstr("duplicate output tensor alias: y"));
}

TEST_F(PredictImplTest, InputTensorsHaveWrongType) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_string_val("any_value");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter(kOutputTensorKey);

  // Input tensors are all wrong.
  absl::Status status = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status.code(),
            static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("to be float but string is provided"));
}

TEST_F(PredictImplTest, ModelMissingSignatures) {
  if (IsTensorflowServingOSS()) {
    return;
  }
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // Model is missing signatures.
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION,
            CallPredict(GetServerCoreWithBadModel(),
                        request, &response).code());
}

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

// Test querying a model with a named regression signature (not default). This
TEST_F(PredictImplTest, PredictionWithNamedRegressionSignature) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  model_spec->set_signature_name("regress_x2_to_y3");

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kRegressInputs] = tensor_proto;
  TF_ASSERT_OK(CallPredict(GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(4);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  *expected_response.mutable_model_spec() = *model_spec;
  (*expected_response.mutable_outputs())[kRegressOutputs] = output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Test querying a model with a classification signature. Predict calls work
// with predict, classify, and regress signatures.
TEST_F(PredictImplTest, PredictionWithNamedClassificationSignature) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  model_spec->set_signature_name("classify_x2_to_y3");

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kClassifyInputs] = tensor_proto;

  TF_ASSERT_OK(CallPredict(GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(4);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  *expected_response.mutable_model_spec() = *model_spec;
  (*expected_response.mutable_outputs())[kClassifyOutputScores] =
      output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Test querying a counter model with signatures. Predict calls work with
// customized signatures. It calls get_counter, incr_counter,
// reset_counter, incr_counter, and incr_counter_by(3) in order.
//
// *Notes*: These signatures are stateful and over-simplied only to demonstrate
// Predict calls with only inputs or outputs. State is not supported in
// TensorFlow Serving on most scalable or production hosting environments.
TEST_F(PredictImplTest, PredictionWithCustomizedSignatures) {
  PredictRequest request;
  PredictResponse response;

  // Call get_counter. Expected result 0.
  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  model_spec->set_signature_name("get_counter");

  TF_ASSERT_OK(CallPredict(GetServerCoreWithCounterModel(),
                           request, &response));

  PredictResponse expected_get_counter;
  *expected_get_counter.mutable_model_spec() = *model_spec;
  TensorProto output_get_counter;
  output_get_counter.add_float_val(0);
  output_get_counter.set_dtype(tensorflow::DT_FLOAT);
  output_get_counter.mutable_tensor_shape();
  (*expected_get_counter.mutable_outputs())["output"] = output_get_counter;
  EXPECT_THAT(response, test_util::EqualsProto(expected_get_counter));

  // Call incr_counter. Expect: 1.
  model_spec->set_signature_name("incr_counter");
  TF_ASSERT_OK(CallPredict(GetServerCoreWithCounterModel(),
                           request, &response));

  PredictResponse expected_incr_counter;
  *expected_incr_counter.mutable_model_spec() = *model_spec;
  TensorProto output_incr_counter;
  output_incr_counter.add_float_val(1);
  output_incr_counter.set_dtype(tensorflow::DT_FLOAT);
  output_incr_counter.mutable_tensor_shape();
  (*expected_incr_counter.mutable_outputs())["output"] = output_incr_counter;
  EXPECT_THAT(response, test_util::EqualsProto(expected_incr_counter));

  // Call reset_counter. Expect: 0.
  model_spec->set_signature_name("reset_counter");
  TF_ASSERT_OK(CallPredict(GetServerCoreWithCounterModel(),
                           request, &response));

  PredictResponse expected_reset_counter;
  *expected_reset_counter.mutable_model_spec() = *model_spec;
  TensorProto output_reset_counter;
  output_reset_counter.add_float_val(0);
  output_reset_counter.set_dtype(tensorflow::DT_FLOAT);
  output_reset_counter.mutable_tensor_shape();
  (*expected_reset_counter.mutable_outputs())["output"] = output_reset_counter;
  EXPECT_THAT(response, test_util::EqualsProto(expected_reset_counter));

  // Call incr_counter. Expect: 1.
  model_spec->set_signature_name("incr_counter");
  request.add_output_filter("output");
  TF_ASSERT_OK(CallPredict(GetServerCoreWithCounterModel(),
                           request, &response));
  request.clear_output_filter();

  PredictResponse expected_incr_counter2;
  *expected_incr_counter2.mutable_model_spec() = *model_spec;
  TensorProto output_incr_counter2;
  output_incr_counter2.add_float_val(1);
  output_incr_counter2.set_dtype(tensorflow::DT_FLOAT);
  output_incr_counter2.mutable_tensor_shape();
  (*expected_incr_counter2.mutable_outputs())["output"] = output_incr_counter2;
  EXPECT_THAT(response, test_util::EqualsProto(expected_incr_counter2));

  // Call incr_counter_by. Expect: 4.
  model_spec->set_signature_name("incr_counter_by");
  TensorProto tensor_proto;
  tensor_proto.add_float_val(3);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())["delta"] = tensor_proto;

  TF_ASSERT_OK(CallPredict(GetServerCoreWithCounterModel(),
                           request, &response));

  PredictResponse expected_incr_counter_by;
  *expected_incr_counter_by.mutable_model_spec() = *model_spec;
  TensorProto output_incr_counter_by;
  output_incr_counter_by.add_float_val(4);
  output_incr_counter_by.set_dtype(tensorflow::DT_FLOAT);
  output_incr_counter_by.mutable_tensor_shape();
  (*expected_incr_counter_by.mutable_outputs())["output"] =
      output_incr_counter_by;
  EXPECT_THAT(response, test_util::EqualsProto(expected_incr_counter_by));
}

TEST_F(PredictImplTest, ThreadPoolOptions) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  test_util::CountingThreadPool inter_op_threadpool(Env::Default(), "InterOp",
                                                    /*num_threads=*/1);
  test_util::CountingThreadPool intra_op_threadpool(Env::Default(), "IntraOp",
                                                    /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options;
  thread_pool_options.inter_op_threadpool = &inter_op_threadpool;
  thread_pool_options.intra_op_threadpool = &intra_op_threadpool;
  TF_EXPECT_OK(
      CallPredict(GetServerCore(), request, &response, thread_pool_options));
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

  // The intra_op_threadpool doesn't have anything scheduled.
  ASSERT_GE(inter_op_threadpool.NumScheduled(), 1);
}

TEST_F(PredictImplTest, MethodNameCheck) {
  ServableHandle<SavedModelBundle> bundle;
  TF_ASSERT_OK(GetSavedModelServableHandle(GetServerCore(), &bundle));
  MetaGraphDef meta_graph_def = bundle->meta_graph_def;
  auto* signature_defs = meta_graph_def.mutable_signature_def();

  PredictRequest request;
  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  FakeSession fake_session;
  PredictResponse response;

  bool old_val = GetSignatureMethodNameCheckFeature();

  SetSignatureMethodNameCheckFeature(true);
  // Legit method names.
  for (const char* method_name :
       {kPredictMethodName, kClassifyMethodName, kRegressMethodName}) {
    (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
        method_name);
    TF_EXPECT_OK(RunPredict(GetRunOptions(), meta_graph_def, kTestModelVersion,
                            &fake_session, request, &response,
                            thread::ThreadPoolOptions()));
  }
  // Unsupported method name will fail check.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      "not/supported/method");
  EXPECT_FALSE(RunPredict(GetRunOptions(), meta_graph_def, kTestModelVersion,
                          &fake_session, request, &response,
                          thread::ThreadPoolOptions())
                   .ok());

  SetSignatureMethodNameCheckFeature(false);
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      kPredictMethodName);
  TF_EXPECT_OK(RunPredict(GetRunOptions(), meta_graph_def, kTestModelVersion,
                          &fake_session, request, &response,
                          thread::ThreadPoolOptions()));
  // Unsupported method name should also work.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      "not/supported/method");
  TF_EXPECT_OK(RunPredict(GetRunOptions(), meta_graph_def, kTestModelVersion,
                          &fake_session, request, &response,
                          thread::ThreadPoolOptions()));

  SetSignatureMethodNameCheckFeature(old_val);
}

SignatureDef CreateMultiOutputSignature(int num_inputs, int num_outputs) {
  SignatureDef signature;
  signature.set_method_name(kPredictMethodName);
  for (int i = 0; i < num_inputs; ++i) {
    TensorInfo input_info;
    input_info.set_name(absl::StrCat("input_tensor_", i, ":0"));
    input_info.set_dtype(DT_FLOAT);
    (*signature.mutable_inputs())[absl::StrCat("input_", i)] = input_info;
  }
  for (int i = 0; i < num_outputs; ++i) {
    TensorInfo output_info;
    output_info.set_name(absl::StrCat("output_tensor_", i, ":0"));
    output_info.set_dtype(DT_FLOAT);
    (*signature.mutable_outputs())[absl::StrCat("output_", i)] = output_info;
  }
  return signature;
}

PredictRequest CreateMultiOutputPredictRequest(int num_inputs,
                                               int num_output_filters) {
  PredictRequest request;
  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  for (int i = 0; i < num_inputs; ++i) {
    TensorProto tensor_proto;
    tensor_proto.add_float_val(static_cast<float>(i));
    tensor_proto.set_dtype(DT_FLOAT);
    tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
    (*request.mutable_inputs())[absl::StrCat("input_", i)] = tensor_proto;
  }
  for (int i = 0; i < num_output_filters; ++i) {
    request.add_output_filter(absl::StrCat("output_", i));
  }
  return request;
}

TEST_F(PredictImplTest, PreProcessPredictionOutputFilterSizes) {
  const SignatureDef signature =
      CreateMultiOutputSignature(/*num_inputs=*/4, /*num_outputs=*/16);

  // Empty output_filter fetches all 16 outputs.
  {
    const PredictRequest request = CreateMultiOutputPredictRequest(
        /*num_inputs=*/4, /*num_output_filters=*/0);
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_ASSERT_OK(internal::PreProcessPrediction(signature, request, &inputs,
                                                &output_tensor_names,
                                                &output_tensor_aliases));
    EXPECT_EQ(inputs.size(), 4);
    EXPECT_EQ(output_tensor_names.size(), 16);
    EXPECT_EQ(output_tensor_aliases.size(), 16);
  }

  // Empty output_filter with 0 signature outputs.
  {
    const SignatureDef empty_outputs_signature =
        CreateMultiOutputSignature(/*num_inputs=*/1, /*num_outputs=*/0);
    const PredictRequest request = CreateMultiOutputPredictRequest(
        /*num_inputs=*/1, /*num_output_filters=*/0);
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_ASSERT_OK(internal::PreProcessPrediction(
        empty_outputs_signature, request, &inputs, &output_tensor_names,
        &output_tensor_aliases));
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_TRUE(output_tensor_names.empty());
    EXPECT_TRUE(output_tensor_aliases.empty());
  }

  // Signature with defaults when fewer inputs are provided in the request, and
  // GetInputValues error when input count matches but an input key is unknown.
  {
    SignatureDef signature_with_defaults =
        CreateMultiOutputSignature(/*num_inputs=*/2, /*num_outputs=*/1);
    TensorProto default_tensor;
    default_tensor.add_float_val(1.0f);
    default_tensor.set_dtype(DT_FLOAT);
    default_tensor.mutable_tensor_shape()->add_dim()->set_size(1);
    (*signature_with_defaults.mutable_defaults())["input_1"] = default_tensor;

    const PredictRequest partial_request = CreateMultiOutputPredictRequest(
        /*num_inputs=*/1, /*num_output_filters=*/0);
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_ASSERT_OK(internal::PreProcessPrediction(
        signature_with_defaults, partial_request, &inputs, &output_tensor_names,
        &output_tensor_aliases));
    EXPECT_EQ(inputs.size(), 2);

    PredictRequest bad_key_request = CreateMultiOutputPredictRequest(
        /*num_inputs=*/1, /*num_output_filters=*/0);
    (*bad_key_request.mutable_inputs())["unknown_input"] = default_tensor;
    inputs.clear();
    output_tensor_names.clear();
    output_tensor_aliases.clear();
    EXPECT_EQ(internal::PreProcessPrediction(
                  signature_with_defaults, bad_key_request, &inputs,
                  &output_tensor_names, &output_tensor_aliases)
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }

  // Valid small (<= 8) and large (> 8) output_filter sizes.
  for (const int num_filters : {1, 4, 8, 9, 12, 16}) {
    const PredictRequest request =
        CreateMultiOutputPredictRequest(/*num_inputs=*/4, num_filters);
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_ASSERT_OK(internal::PreProcessPrediction(signature, request, &inputs,
                                                &output_tensor_names,
                                                &output_tensor_aliases));
    ASSERT_EQ(output_tensor_names.size(), num_filters);
    ASSERT_EQ(output_tensor_aliases.size(), num_filters);
    for (int i = 0; i < num_filters; ++i) {
      EXPECT_EQ(output_tensor_names[i],
                absl::StrCat("output_tensor_", i, ":0"));
      EXPECT_EQ(output_tensor_aliases[i], absl::StrCat("output_", i));
    }
  }

  // Pre-populated out-vectors: duplicate detection only considers aliases
  // appended by this call, so "output_0" is not rejected against the existing
  // "output_0" entry.
  {
    const PredictRequest request = CreateMultiOutputPredictRequest(
        /*num_inputs=*/4, /*num_output_filters=*/3);
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names = {"stale"};
    std::vector<std::string> output_tensor_aliases = {"output_0"};
    TF_ASSERT_OK(internal::PreProcessPrediction(signature, request, &inputs,
                                                &output_tensor_names,
                                                &output_tensor_aliases));
    EXPECT_THAT(
        output_tensor_aliases,
        ::testing::ElementsAre("output_0", "output_0", "output_1", "output_2"));
    EXPECT_THAT(
        output_tensor_names,
        ::testing::ElementsAre("stale", "output_tensor_0:0",
                               "output_tensor_1:0", "output_tensor_2:0"));
  }

  // Duplicate output_filter alias rejection for both <= 8 and > 8 filters.
  for (const int num_filters : {3, 10}) {
    PredictRequest request =
        CreateMultiOutputPredictRequest(/*num_inputs=*/4, num_filters);
    request.add_output_filter("output_1");
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    const absl::Status status = internal::PreProcessPrediction(
        signature, request, &inputs, &output_tensor_names,
        &output_tensor_aliases);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(
        status.message(),
        ::testing::HasSubstr("duplicate output tensor alias: output_1"));
  }

  // Unknown output_filter alias rejection for both <= 8 and > 8 filters.
  for (const int num_filters : {3, 10}) {
    PredictRequest request =
        CreateMultiOutputPredictRequest(/*num_inputs=*/4, num_filters);
    request.add_output_filter("non_existent_output");
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    const absl::Status status = internal::PreProcessPrediction(
        signature, request, &inputs, &output_tensor_names,
        &output_tensor_aliases);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(status.message(),
                ::testing::HasSubstr("output tensor alias not found in "
                                     "signature: non_existent_output"));
  }
}

TEST_F(PredictImplTest, PostProcessPredictionResultOptionsAndErrors) {
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_signature_def())[kDefaultServingSignatureDefKey] =
      CreateMultiOutputSignature(/*num_inputs=*/1, /*num_outputs=*/1);
  const PredictRequest request = CreateMultiOutputPredictRequest(
      /*num_inputs=*/1, /*num_output_filters=*/1);
  FakeSession fake_session;
  PredictResponse response;
  TF_ASSERT_OK(internal::RunPredict(
      GetRunOptions(), meta_graph_def, kTestModelVersion,
      internal::PredictResponseTensorSerializationOption::kAsProtoContent,
      &fake_session, request, &response));
  ASSERT_TRUE(response.outputs().contains("output_0"));
  EXPECT_FALSE(response.outputs().at("output_0").tensor_content().empty());

  PredictResponse empty_response;
  TF_EXPECT_OK(internal::PostProcessPredictionResult(
      /*output_tensor_aliases=*/{}, /*output_tensors=*/{},
      internal::PredictResponseTensorSerializationOption::kAsProtoContent,
      &empty_response));

  const absl::Status mismatch_status = internal::PostProcessPredictionResult(
      /*output_tensor_aliases=*/{"output_0"}, /*output_tensors=*/{},
      internal::PredictResponseTensorSerializationOption::kAsProtoField,
      &empty_response);
  EXPECT_EQ(mismatch_status.code(), absl::StatusCode::kUnknown);
}

void BM_PreProcessPrediction(
    ::testing::benchmark::State& state) {  // NOLINT(misc-include-cleaner)
  const int num_output_filters = state.range(0);
  const int num_outputs = num_output_filters > 0 ? num_output_filters : 8;
  const SignatureDef signature =
      CreateMultiOutputSignature(/*num_inputs=*/4, num_outputs);
  const PredictRequest request =
      CreateMultiOutputPredictRequest(/*num_inputs=*/4, num_output_filters);
  for (auto s : state) {
    std::vector<std::pair<std::string, Tensor>> inputs;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_CHECK_OK(internal::PreProcessPrediction(signature, request, &inputs,
                                               &output_tensor_names,
                                               &output_tensor_aliases));
    tensorflow::testing::DoNotOptimize(inputs);
    tensorflow::testing::DoNotOptimize(output_tensor_names);
    tensorflow::testing::DoNotOptimize(output_tensor_aliases);
  }
}
BENCHMARK(BM_PreProcessPrediction)  // NOLINT(misc-include-cleaner)
    ->Arg(0)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

void BM_RunPredict(
    ::testing::benchmark::State& state) {  // NOLINT(misc-include-cleaner)
  const int num_output_filters = state.range(0);
  const int num_outputs = num_output_filters > 0 ? num_output_filters : 8;
  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_signature_def())[kDefaultServingSignatureDefKey] =
      CreateMultiOutputSignature(/*num_inputs=*/4, num_outputs);
  const PredictRequest request =
      CreateMultiOutputPredictRequest(/*num_inputs=*/4, num_output_filters);
  FakeSession fake_session;
  const RunOptions run_options;
  for (auto s : state) {
    PredictResponse response;
    TF_CHECK_OK(RunPredict(run_options, meta_graph_def, kTestModelVersion,
                           &fake_session, request, &response));
    tensorflow::testing::DoNotOptimize(response);
  }
}
BENCHMARK(BM_RunPredict)  // NOLINT(misc-include-cleaner)
    ->Arg(0)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

}  // namespace
}  // namespace serving
}  // namespace tensorflow
