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

#include "tensorflow_serving/servables/tensorflow/tfrt_classification_service.h"

#include <memory>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";

// Test fixture for ClassificationService related tests sets up a ServerCore
// pointing to the half_plus_two SavedModel.
class TFRTClassificationServiceTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));

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

  static void TearDownTestSuite() { server_core_ = nullptr; }

 protected:
  static std::unique_ptr<ServerCore> server_core_;
  Servable::RunOptions run_options_;
};

std::unique_ptr<ServerCore> TFRTClassificationServiceTest::server_core_;

// Verifies that Classify() returns an error for different cases of an invalid
// ClassificationRequest.model_spec.
TEST_F(TFRTClassificationServiceTest, InvalidModelSpec) {
  ClassificationRequest request;
  ClassificationResponse response;

  // No model_spec specified.
  EXPECT_EQ(TFRTClassificationServiceImpl::Classify(
                run_options_, server_core_.get(), request, &response)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // No model name specified.
  auto* model_spec = request.mutable_model_spec();
  EXPECT_EQ(TFRTClassificationServiceImpl::Classify(
                run_options_, server_core_.get(), request, &response)
                .code(),
            absl::StatusCode::kInvalidArgument);

  // No servable found for model name "foo".
  model_spec->set_name("foo");
  EXPECT_EQ(TFRTClassificationServiceImpl::Classify(
                run_options_, server_core_.get(), request, &response)
                .code(),
            tensorflow::error::NOT_FOUND);
}

// Verifies that Classify() returns an error for an invalid signature_name in
// ClassificationRequests's model_spec.
TEST_F(TFRTClassificationServiceTest, InvalidSignature) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "  signature_name: \"invalid_signature_name\""
      "}");
  ClassificationResponse response;
  EXPECT_EQ(TFRTClassificationServiceImpl::Classify(
                run_options_, server_core_.get(), request, &response)
                .code(),
            tensorflow::error::FAILED_PRECONDITION);
}

// Verifies that Classify() returns the correct score for a valid
// ClassificationRequest against the half_plus_two SavedModel's classify_x_to_y
// signature.
TEST_F(TFRTClassificationServiceTest, ClassificationSuccess) {
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
      "  }"
      "}");
  ClassificationResponse response;
  TF_EXPECT_OK(TFRTClassificationServiceImpl::Classify(
      run_options_, server_core_.get(), request, &response));
  EXPECT_THAT(response,
              test_util::EqualsProto(
                  "result { classifications { classes { score: 42 } } }"
                  "model_spec {"
                  "  name: \"test_model\""
                  "  signature_name: \"classify_x_to_y\""
                  "  version { value: 123 }"
                  "}"));
}

// Verifies that ClassifyWithModelSpec() uses the model spec override rather
// than the one in the request.
TEST_F(TFRTClassificationServiceTest, ModelSpecOverride) {
  auto request = test_util::CreateProto<ClassificationRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "}");
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  ClassificationResponse response;
  EXPECT_NE(tensorflow::error::NOT_FOUND,
            TFRTClassificationServiceImpl::Classify(
                run_options_, server_core_.get(), request, &response)
                .code());
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            TFRTClassificationServiceImpl::ClassifyWithModelSpec(
                run_options_, server_core_.get(), model_spec_override, request,
                &response)
                .code());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
