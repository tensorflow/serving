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

#include "tensorflow_serving/model_servers/get_model_status_impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/status.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelBasePath[] =
    "/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions";
constexpr char kTestModelName[] = "saved_model_half_plus_two_2_versions";
constexpr char kNonexistentModelName[] = "nonexistent_model";
constexpr int kTestModelVersion1 = 123;
constexpr int kTestModelVersion2 = 124;
constexpr int kNonexistentModelVersion = 125;

class GetModelStatusImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    TF_ASSERT_OK(CreateServerCore(&server_core_));
  }

  static void TearDownTestSuite() { server_core_.reset(); }

 protected:
  static Status CreateServerCore(std::unique_ptr<ServerCore>* server_core) {
    ModelServerConfig config;
    auto* model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(test_util::TestSrcDirPath(kTestModelBasePath));
    auto* specific_versions =
        model_config->mutable_model_version_policy()->mutable_specific();
    specific_versions->add_versions(kTestModelVersion1);
    specific_versions->add_versions(kTestModelVersion2);

    model_config->set_model_platform(kTensorFlowModelPlatform);

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;

    options.platform_config_map =
        CreateTensorFlowPlatformConfigMap(SessionBundleConfig());
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    return ServerCore::Create(std::move(options), server_core);
  }

  ServerCore* GetServerCore() { return server_core_.get(); }

 private:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> GetModelStatusImplTest::server_core_;

TEST_F(GetModelStatusImplTest, MissingOrEmptyModelSpecFailure) {
  GetModelStatusRequest request;
  GetModelStatusResponse response;

  // Empty request is invalid.
  EXPECT_EQ(
      tensorflow::error::INVALID_ARGUMENT,
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response)
          .code());
}

TEST_F(GetModelStatusImplTest, InvalidModelNameFailure) {
  GetModelStatusRequest request;
  GetModelStatusResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kNonexistentModelName);

  // If no versions of model are managed by ServerCore, response is
  // NOT_FOUND error.
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response)
          .code());
  EXPECT_EQ(0, response.model_version_status_size());
}

TEST_F(GetModelStatusImplTest, InvalidModelVersionFailure) {
  GetModelStatusRequest request;
  GetModelStatusResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kNonexistentModelVersion);

  // If model version not managed by ServerCore, response is NOT_FOUND error.
  EXPECT_EQ(
      tensorflow::error::NOT_FOUND,
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response)
          .code());
  EXPECT_EQ(0, response.model_version_status_size());
}

TEST_F(GetModelStatusImplTest, AllVersionsSuccess) {
  GetModelStatusRequest request;
  GetModelStatusResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);

  // If two versions of model are managed by ServerCore, succesfully get model
  // status for both versions of the model.
  TF_EXPECT_OK(
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response));
  EXPECT_EQ(2, response.model_version_status_size());
  std::set<int64_t> expected_versions = {kTestModelVersion1,
                                         kTestModelVersion2};
  std::set<int64_t> actual_versions = {
      response.model_version_status(0).version(),
      response.model_version_status(1).version()};
  EXPECT_EQ(expected_versions, actual_versions);
  EXPECT_EQ(tensorflow::error::OK,
            response.model_version_status(0).status().error_code());
  EXPECT_EQ("", response.model_version_status(0).status().error_message());
  EXPECT_EQ(tensorflow::error::OK,
            response.model_version_status(1).status().error_code());
  EXPECT_EQ("", response.model_version_status(1).status().error_message());
}

TEST_F(GetModelStatusImplTest, SingleVersionSuccess) {
  GetModelStatusRequest request;
  GetModelStatusResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion1);

  // If model version is managed by ServerCore, succesfully get model status.
  TF_EXPECT_OK(
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response));
  EXPECT_EQ(1, response.model_version_status_size());
  EXPECT_EQ(kTestModelVersion1, response.model_version_status(0).version());
  EXPECT_EQ(tensorflow::error::OK,
            response.model_version_status(0).status().error_code());
  EXPECT_EQ("", response.model_version_status(0).status().error_message());
}

// Verifies that GetModelStatusWithModelSpec() uses the model spec override
// rather than the one in the request.
TEST_F(GetModelStatusImplTest, ModelSpecOverride) {
  GetModelStatusRequest request;
  request.mutable_model_spec()->set_name(kTestModelName);
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  GetModelStatusResponse response;
  EXPECT_NE(
      tensorflow::error::NOT_FOUND,
      GetModelStatusImpl::GetModelStatus(GetServerCore(), request, &response)
          .code());
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            GetModelStatusImpl::GetModelStatusWithModelSpec(
                GetServerCore(), model_spec_override, request, &response)
                .code());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
