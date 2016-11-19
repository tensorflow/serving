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
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/eager_load_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

class PredictImplTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    TF_ASSERT_OK(CreateServerCore(
        "/servables/tensorflow/testdata/half_plus_two", &server_core_));
    TF_ASSERT_OK(
        CreateServerCore("/servables/tensorflow/testdata/bad_half_plus_two",
                         &server_core_bad_model_));
  }

  static void TearDownTestCase() {
    server_core_.reset();
    server_core_bad_model_.reset();
  }

 protected:
  static Status CreateServerCore(const string& model_path,
                                 std::unique_ptr<ServerCore>* server_core) {
    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;

    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(test_util::TestSrcDirPath(model_path));
    model_config->set_model_platform(kTensorFlowModelPlatform);
    options.model_server_config = config;

    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new EagerLoadPolicy);
    return ServerCore::Create(std::move(options), server_core);
  }

  static std::unique_ptr<ServerCore> server_core_;
  static std::unique_ptr<ServerCore> server_core_bad_model_;
};

std::unique_ptr<ServerCore> PredictImplTest::server_core_;
std::unique_ptr<ServerCore> PredictImplTest::server_core_bad_model_;

TEST_F(PredictImplTest, MissingOrEmptyModelSpec) {
  PredictRequest request;
  PredictResponse response;

  // Empty request is invalid.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());

  // Model name is wrong, not found.
  model_spec->set_name("test");
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());
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
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
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

  auto inputs = request.mutable_inputs();
  (*inputs)["key"] = tensor_proto;
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());
}

TEST_F(PredictImplTest, OutputFiltersDontMatchModelSpecOutputs) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  ServableHandle<SessionBundle> bundle;
  TF_ASSERT_OK(server_core_->GetServableHandle(request.model_spec(), &bundle));
  Signature signature;
  TF_ASSERT_OK(GetNamedSignature("inputs", bundle->meta_graph_def, &signature));

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);

  for (const auto& input : signature.generic_signature().map()) {
    (*request.mutable_inputs())[input.first] = tensor_proto;
  }

  request.add_output_filter("output_filter");

  // Output filter like this doesn't exist.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());

  request.clear_output_filter();
  request.add_output_filter("y");
  EXPECT_TRUE(
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .ok());
  request.add_output_filter("y");

  // Duplicate output filter specified.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());
}

TEST_F(PredictImplTest, InputTensorsHaveWrongType) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  ServableHandle<SessionBundle> bundle;
  TF_ASSERT_OK(server_core_->GetServableHandle(request.model_spec(), &bundle));

  TensorProto tensor_proto;
  tensor_proto.add_string_val("any_key");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);

  Signature signature;
  TF_ASSERT_OK(GetNamedSignature("inputs", bundle->meta_graph_def, &signature));
  for (const auto& input : signature.generic_signature().map()) {
    (*request.mutable_inputs())[input.first] = tensor_proto;
  }
  request.add_output_filter("y");
  // Input tensors are all wrong.
  EXPECT_EQ(
      tensorflow::error::INTERNAL,
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .code());
}

TEST_F(PredictImplTest, ModelMissingSignatures) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  // Model is missing signatures.
  EXPECT_EQ(tensorflow::error::FAILED_PRECONDITION,
            TensorflowPredictImpl::Predict(server_core_bad_model_.get(),
                                           request, &response)
                .code());
}

TEST_F(PredictImplTest, PredictionSuccess) {
  PredictRequest request;
  PredictResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);

  ServableHandle<SessionBundle> bundle;
  TF_ASSERT_OK(server_core_->GetServableHandle(request.model_spec(), &bundle));
  Signature signature;
  TF_ASSERT_OK(GetNamedSignature("inputs", bundle->meta_graph_def, &signature));

  TensorProto tensor_proto;
  tensor_proto.add_float_val(2.0);
  tensor_proto.set_dtype(tensorflow::DT_FLOAT);

  for (const auto& input : signature.generic_signature().map()) {
    (*request.mutable_inputs())[input.first] = tensor_proto;
  }

  EXPECT_TRUE(
      TensorflowPredictImpl::Predict(server_core_.get(), request, &response)
          .ok());
  TensorProto output_tensor_proto;
  output_tensor_proto.add_float_val(3);
  output_tensor_proto.set_dtype(tensorflow::DT_FLOAT);
  output_tensor_proto.mutable_tensor_shape();
  PredictResponse test_response;
  (*test_response.mutable_outputs())["y"] = output_tensor_proto;
  EXPECT_THAT(test_response, test_util::EqualsProto(response));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
