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
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

const char kInputTensorKey[] = "x";
const char kOutputTensorKey[] = "y";

// Parameter is 'bool use_saved_model'.
class PredictImplTest : public ::testing::TestWithParam<bool> {
 public:
  static void SetUpTestCase() {
    TF_ASSERT_OK(
        CreateServerCore(test_util::TestSrcDirPath(
                             "/servables/tensorflow/testdata/half_plus_two"),
                         false, &server_core_));
    const string bad_half_plus_two_path = test_util::TestSrcDirPath(
        "/servables/tensorflow/testdata/bad_half_plus_two");
    TF_ASSERT_OK(CreateServerCore(bad_half_plus_two_path, false,
                                  &server_core_bad_model_));

    TF_ASSERT_OK(CreateServerCore(test_util::TensorflowTestSrcDirPath(
                                      "cc/saved_model/testdata/half_plus_two"),
                                  true, &saved_model_server_core_));
    TF_ASSERT_OK(CreateServerCore(bad_half_plus_two_path, true,
                                  &saved_model_server_core_bad_model_));
  }

  static void TearDownTestCase() {
    server_core_.reset();
    server_core_bad_model_.reset();
    saved_model_server_core_.reset();
    saved_model_server_core_bad_model_.reset();
  }

 protected:
  static Status CreateServerCore(const string& model_path, bool use_saved_model,
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
    options.platform_config_map = CreateTensorFlowPlatformConfigMap(
        SessionBundleConfig(), use_saved_model);
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    return ServerCore::Create(std::move(options), server_core);
  }

  ServerCore* GetServerCore() {
    if (GetParam()) {
      return saved_model_server_core_.get();
    }
    return server_core_.get();
  }

  ServerCore* GetServerCoreWithBadModel() {
    if (GetParam()) {
      return saved_model_server_core_bad_model_.get();
    }
    return server_core_bad_model_.get();
  }

  RunOptions GetRunOptions() { return RunOptions(); }

 private:
  static std::unique_ptr<ServerCore> server_core_;
  static std::unique_ptr<ServerCore> server_core_bad_model_;
  static std::unique_ptr<ServerCore> saved_model_server_core_;
  static std::unique_ptr<ServerCore> saved_model_server_core_bad_model_;
};

std::unique_ptr<ServerCore> PredictImplTest::server_core_;
std::unique_ptr<ServerCore> PredictImplTest::server_core_bad_model_;
std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_;
std::unique_ptr<ServerCore> PredictImplTest::saved_model_server_core_bad_model_;

TEST_P(PredictImplTest, MissingOrEmptyModelSpec) {
  PredictRequest request;
  PredictResponse response;

  // Empty request is invalid.
  TensorflowPredictor predictor(GetParam());
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

TEST_P(PredictImplTest, EmptyInputList) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorflowPredictor predictor(GetParam());
  // The input is empty.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_P(PredictImplTest, InputTensorsDontMatchModelSpecInputs) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_string_val("any_key");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);

  TensorflowPredictor predictor(GetParam());
  auto inputs = request.mutable_inputs();
  (*inputs)["key"] = tensor_proto;
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_P(PredictImplTest, OutputFiltersDontMatchModelSpecOutputs) {
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

  TensorflowPredictor predictor(GetParam());
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

TEST_P(PredictImplTest, InputTensorsHaveWrongType) {
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

  TensorflowPredictor predictor(GetParam());
  // Input tensors are all wrong.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
          .code());
}

TEST_P(PredictImplTest, ModelMissingSignatures) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // Model is missing signatures.
  TensorflowPredictor predictor(GetParam());
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION,
            predictor
                .Predict(GetRunOptions(), GetServerCoreWithBadModel(), request,
                         &response)
                .code());
}

TEST_P(PredictImplTest, PredictionSuccess) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  (*request.mutable_inputs())[kInputTensorKey] = tensor_proto;

  TensorflowPredictor predictor(GetParam());
  TF_EXPECT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(3);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  (*expected_response.mutable_outputs())[kOutputTensorKey] =
      output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Test querying a model with a named regression signature (not default). This
// will work with SavedModel but not supported in the legacy SessionBundle.
TEST_P(PredictImplTest, PredictionWithNamedRegressionSignature) {
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
  TensorflowPredictor predictor(GetParam());
  // This request is expected to work with SavedModel, but not SessionBundle.
  if (GetParam()) {
    TF_ASSERT_OK(predictor.Predict(GetRunOptions(), GetServerCore(), request,
                                   &response));
  } else {
    ASSERT_EQ(
        tensorflow::error::INVALID_ARGUMENT,
        predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
            .code());
    return;
  }
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(4);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  (*expected_response.mutable_outputs())[kRegressOutputs] = output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Test querying a model with a classification signature. Predict calls work
// with predict, classify, and regress signatures when using SavedModel, but
// will only work with a generic signature when using legacy SessionBundle.
TEST_P(PredictImplTest, PredictionWithNamedClassificationSignature) {
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

  TensorflowPredictor predictor(GetParam());
  // This request is expected to work with SavedModel, but not SessionBundle.
  const bool using_session_bundle = !GetParam();
  if (using_session_bundle) {
    ASSERT_EQ(
        tensorflow::error::INVALID_ARGUMENT,
        predictor.Predict(GetRunOptions(), GetServerCore(), request, &response)
            .code());
    return;
  }
  TF_ASSERT_OK(
      predictor.Predict(GetRunOptions(), GetServerCore(), request, &response));
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(4);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse expected_response;
  (*expected_response.mutable_outputs())[kClassifyOutputScores] =
      output_tensor_proto;
  EXPECT_THAT(response, test_util::EqualsProto(expected_response));
}

// Test all PredictImplTest test cases with both SessionBundle and SavedModel.
INSTANTIATE_TEST_CASE_P(UseSavedModel, PredictImplTest, ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
