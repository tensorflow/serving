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

#include "tensorflow_serving/servables/tensorflow/regression_service.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";

// Test fixture for RegressionService related tests sets up a ServerCore
// pointing to the half_plus_two SavedModel.
class RegressionServiceTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(test_util::TensorflowTestSrcDirPath(
        "cc/saved_model/testdata/half_plus_two"));
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
    TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core_));
  }

  static void TearDownTestSuite() { server_core_ = nullptr; }

 protected:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> RegressionServiceTest::server_core_;

// Verifies that Regress() returns an error for different cases of an invalid
// RegressionRequest.model_spec.
TEST_F(RegressionServiceTest, InvalidModelSpec) {
  RegressionRequest request;
  RegressionResponse response;

  // No model_spec specified.
  EXPECT_EQ(TensorflowRegressionServiceImpl::Regress(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                request, &response)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);

  // No model name specified.
  auto* model_spec = request.mutable_model_spec();
  EXPECT_EQ(TensorflowRegressionServiceImpl::Regress(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                request, &response)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);

  // No servable found for model name "foo".
  model_spec->set_name("foo");
  EXPECT_EQ(TensorflowRegressionServiceImpl::Regress(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                request, &response)
                .code(),
            tensorflow::error::NOT_FOUND);
}

// Verifies that Regress() returns an error for an invalid signature_name in
// RegressionRequests's model_spec.
TEST_F(RegressionServiceTest, InvalidSignature) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "    name: \"test_model\""
      "    signature_name: \"invalid_signature_name\""
      "}");
  RegressionResponse response;
  EXPECT_EQ(TensorflowRegressionServiceImpl::Regress(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                request, &response)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

// Verifies that Regress() returns the correct value for a valid
// RegressionRequest against the half_plus_two SavedModel's regress_x_to_y
// signature.
TEST_F(RegressionServiceTest, RegressionSuccess) {
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
      "  }"
      "}");
  RegressionResponse response;
  TF_EXPECT_OK(TensorflowRegressionServiceImpl::Regress(
      RunOptions(), server_core_.get(), thread::ThreadPoolOptions(), request,
      &response));
  EXPECT_THAT(response,
              test_util::EqualsProto("result { regressions { value: 42 } }"
                                     "model_spec {"
                                     "  name: \"test_model\""
                                     "  signature_name: \"regress_x_to_y\""
                                     "  version { value: 123 }"
                                     "}"));
}

// Verifies that RegressWithModelSpec() uses the model spec override rather than
// the one in the request.
TEST_F(RegressionServiceTest, ModelSpecOverride) {
  auto request = test_util::CreateProto<RegressionRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "}");
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  RegressionResponse response;
  EXPECT_NE(tensorflow::error::NOT_FOUND,
            TensorflowRegressionServiceImpl::Regress(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                request, &response)
                .code());
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            TensorflowRegressionServiceImpl::RegressWithModelSpec(
                RunOptions(), server_core_.get(), thread::ThreadPoolOptions(),
                model_spec_override, request, &response)
                .code());
}

TEST_F(RegressionServiceTest, ThreadPoolOptions) {
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
      "  }"
      "}");

  test_util::CountingThreadPool inter_op_threadpool(Env::Default(), "InterOp",
                                                    /*num_threads=*/1);
  test_util::CountingThreadPool intra_op_threadpool(Env::Default(), "IntraOp",
                                                    /*num_threads=*/1);
  thread::ThreadPoolOptions thread_pool_options;
  thread_pool_options.inter_op_threadpool = &inter_op_threadpool;
  thread_pool_options.intra_op_threadpool = &intra_op_threadpool;
  RegressionResponse response;
  TF_EXPECT_OK(TensorflowRegressionServiceImpl::Regress(
      RunOptions(), server_core_.get(), thread_pool_options, request,
      &response));
  EXPECT_THAT(response,
              test_util::EqualsProto("result { regressions { value: 42 } }"
                                     "model_spec {"
                                     "  name: \"test_model\""
                                     "  signature_name: \"regress_x_to_y\""
                                     "  version { value: 123 }"
                                     "}"));
  // The intra_op_threadpool doesn't have anything scheduled.
  ASSERT_GE(inter_op_threadpool.NumScheduled(), 1);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
