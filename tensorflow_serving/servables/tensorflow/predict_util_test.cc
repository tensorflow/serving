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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
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
  Status Create(const GraphDef& graph) override {
    return errors::Unimplemented("not available in fake");
  }
  Status Extend(const GraphDef& graph) override {
    return errors::Unimplemented("not available in fake");
  }
  Status Close() override {
    return errors::Unimplemented("not available in fake");
  }
  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return errors::Unimplemented("not available in fake");
  }
  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs) override {
    RunMetadata run_metadata;
    return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
               &run_metadata);
  }
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
    return Run(run_options, inputs, output_names, target_nodes, outputs,
               run_metadata, thread::ThreadPoolOptions());
  }
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override {
    for (const auto& t : inputs) {
      outputs->push_back(t.second);
    }
    return OkStatus();
  }
};

class PredictImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    if (!IsTensorflowServingOSS()) {
      const string bad_half_plus_two_path = test_util::TestSrcDirPath(
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
  static Status CreateServerCore(const string& model_path,
                                 std::unique_ptr<ServerCore>* server_core) {
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

  Status GetSavedModelServableHandle(ServerCore* server_core,
                                     ServableHandle<SavedModelBundle>* bundle) {
    ModelSpec model_spec;
    model_spec.set_name(kTestModelName);
    return server_core->GetServableHandle(model_spec, bundle);
  }

  Status CallPredict(ServerCore* server_core, const PredictRequest& request,
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
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            CallPredict(GetServerCore(), request, &response).code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            CallPredict(GetServerCore(), request, &response).code());

  // Model name is wrong.
  model_spec->set_name("test");
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            CallPredict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, EmptyInputList) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // The input is empty.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
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

  Status status = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("Sent extra: {unknown_key1,unknown_key2}"));
  EXPECT_THAT(status.error_message(),
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
      status.error_message(),
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
  Status status1 = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status1.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status1.error_message(),
              ::testing::HasSubstr(
                  "output tensor alias not found in signature: output_filter"));

  request.clear_output_filter();
  request.add_output_filter(kOutputTensorKey);
  TF_EXPECT_OK(CallPredict(GetServerCore(), request, &response));
  request.add_output_filter(kOutputTensorKey);

  // Duplicate output filter specified.
  Status status2 = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status2.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status2.error_message(),
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
  Status status = CallPredict(GetServerCore(), request, &response);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
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
  // Legit method name.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      kPredictMethodName);
  TF_EXPECT_OK(RunPredict(GetRunOptions(), meta_graph_def, kTestModelVersion,
                          &fake_session, request, &response,
                          thread::ThreadPoolOptions()));
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

}  // namespace
}  // namespace serving
}  // namespace tensorflow
