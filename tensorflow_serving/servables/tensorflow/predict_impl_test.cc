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

#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.h"
#include "tensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.pb.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/oss_or_google.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

const char kInputTensorKey[] = "x";
const char kOutputTensorKey[] = "y";

class PredictImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    TF_ASSERT_OK(CreateServerCore(test_util::TensorflowTestSrcDirPath(
                                      "cc/saved_model/testdata/half_plus_two"),
                                  &saved_model_server_core_));
    TF_ASSERT_OK(CreateServerCore(
        test_util::TestSrcDirPath(
            "/servables/tensorflow/testdata/saved_model_counter"),
        &saved_model_server_core_counter_model_));
  }

  static void TearDownTestSuite() {
    server_core_.reset();
    server_core_bad_model_.reset();
    saved_model_server_core_.reset();
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

  ServerCore* GetServerCore() { return saved_model_server_core_.get(); }

  ServerCore* GetServerCoreWithCounterModel() {
    return saved_model_server_core_counter_model_.get();
  }

  RunOptions GetRunOptions() { return RunOptions(); }

 private:
  static std::unique_ptr<ServerCore> server_core_;
  static std::unique_ptr<ServerCore> server_core_bad_model_;
  static std::unique_ptr<ServerCore> saved_model_server_core_;
  static std::unique_ptr<ServerCore> saved_model_server_core_counter_model_;
};

std::unique_ptr<ServerCore> PredictImplTest::server_core_;
std::unique_ptr<ServerCore> PredictImplTest::server_core_bad_model_;
std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_;
std::unique_ptr<ServerCore>
    PredictImplTest::saved_model_server_core_counter_model_;

TEST_F(PredictImplTest, MissingOrEmptyModelSpec) {
  PredictRequest request;
  PredictResponse response;

  // Empty request is invalid.
  TensorflowPredictor predictor;
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());

  // Model name is wrong, not found.
  model_spec->set_name("test");
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_F(PredictImplTest, EmptyInputList) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorflowPredictor predictor;
  // The input is empty.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_F(PredictImplTest, InputTensorsDontMatchModelSpecInputs) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_string_val("any_key");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);

  TensorflowPredictor predictor;
  auto inputs = request.mutable_inputs();
  (*inputs)["key"] = tensor_proto;
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
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

  TensorflowPredictor predictor;
  // Output filter like this doesn't exist.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());

  request.clear_output_filter();
  request.add_output_filter(kOutputTensorKey);
  TF_EXPECT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
  request.add_output_filter(kOutputTensorKey);

  // Duplicate output filter specified.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_F(PredictImplTest, InputTensorsHaveWrongType) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_string_val("any_key");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;
  request.add_output_filter(kOutputTensorKey);

  TensorflowPredictor predictor;
  // Input tensors are all wrong.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
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

  TensorflowPredictor predictor;
  TF_EXPECT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
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

// Test querying a model with a named regression signature (not default).
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
  TensorflowPredictor predictor;
  TF_ASSERT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
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
// with predict, classify, and regress signatures when using SavedModel.
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

  TensorflowPredictor predictor;
  TF_ASSERT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
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
  TensorflowPredictor predictor;

  // Call get_counter. Expected result 0.
  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  model_spec->set_signature_name("get_counter");

  TF_ASSERT_OK(predictor.Predict(
      GetRunOptions(), GetServerCoreWithCounterModel(), request, &response));

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
  TF_ASSERT_OK(predictor.Predict(
      GetRunOptions(), GetServerCoreWithCounterModel(), request, &response));

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
  TF_ASSERT_OK(predictor.Predict(
      GetRunOptions(), GetServerCoreWithCounterModel(), request, &response));

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
  TF_ASSERT_OK(predictor.Predict(
      GetRunOptions(), GetServerCoreWithCounterModel(), request, &response));
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

  TF_ASSERT_OK(predictor.Predict(
      GetRunOptions(), GetServerCoreWithCounterModel(), request, &response));

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

// Verifies that PredictWithModelSpec() uses the model spec override rather than
// the one in the request.
TEST_F(PredictImplTest, ModelSpecOverride) {
  auto request = test_util::CreateProto<PredictRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "}");
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  TensorflowPredictor predictor;
  PredictResponse response;
  EXPECT_NE(tensorflow::error::NOT_FOUND,
            predictor.Predict(RunOptions(), GetServerCore(), request, &response)
                .code());
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            predictor
                .PredictWithModelSpec(RunOptions(), GetServerCore(),
                                      model_spec_override, request, &response)
                .code());
}

TEST_F(PredictImplTest, ThreadPoolFactory) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  auto inter_op_threadpool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "InterOp",
                                                      /*num_threads=*/1);
  auto intra_op_threadpool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "IntraOp",
                                                      /*num_threads=*/1);
  test_util::FakeThreadPoolFactoryConfig fake_thread_pool_factory_config;
  test_util::FakeThreadPoolFactory fake_thread_pool_factory(
      fake_thread_pool_factory_config);
  fake_thread_pool_factory.SetInterOpThreadPool(inter_op_threadpool);
  fake_thread_pool_factory.SetIntraOpThreadPool(intra_op_threadpool);

  TensorflowPredictor predictor(&fake_thread_pool_factory);
  TF_EXPECT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
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
  ASSERT_GE(inter_op_threadpool->NumScheduled(), 1);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
