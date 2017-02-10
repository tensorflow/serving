/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/wrappers.pb.h"
#include "google/protobuf/map.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
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
const string kSignatureDef = "signature_def";

class GetModelMetadataImplTest : public ::testing::TestWithParam<bool> {
 public:
  static void SetUpTestCase() {
    const string session_bundle_path = test_util::TestSrcDirPath(
        "/servables/tensorflow/testdata/half_plus_two");
    TF_ASSERT_OK(CreateServerCore(session_bundle_path, false, &server_core_));

    const string saved_model_path = test_util::TensorflowTestSrcDirPath(
        "cc/saved_model/testdata/half_plus_two");
    TF_ASSERT_OK(
        CreateServerCore(saved_model_path, true, &saved_model_server_core_));
  }

  static void TearDownTestCase() {
    server_core_.reset();
    saved_model_server_core_.reset();
  }

 protected:
  static Status CreateServerCore(const string& model_path,
                                 bool saved_model_on_disk,
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
        CreateTensorFlowPlatformConfigMap(SessionBundleConfig(), true);
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

 private:
  static std::unique_ptr<ServerCore> server_core_;
  static std::unique_ptr<ServerCore> saved_model_server_core_;
};

std::unique_ptr<ServerCore> GetModelMetadataImplTest::server_core_;
std::unique_ptr<ServerCore> GetModelMetadataImplTest::saved_model_server_core_;

SignatureDefMap GetSignatureDefMap(ServerCore* server_core,
                                   const ModelSpec& model_spec) {
  SignatureDefMap signature_def_map;
  ServableHandle<SavedModelBundle> bundle;
  TF_EXPECT_OK(server_core->GetServableHandle(model_spec, &bundle));
  for (const auto& signature : bundle->meta_graph_def.signature_def()) {
    (*signature_def_map.mutable_signature_def())[signature.first] =
        signature.second;
  }
  return signature_def_map;
}

TEST_P(GetModelMetadataImplTest, EmptyOrInvalidMetadataFieldList) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  // Empty metadata field list is invalid.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                   &response)
                .code());
  request.add_metadata_field("some_stuff");

  // Field enum is outside of valid range.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                   &response)
                .code());
}

TEST_P(GetModelMetadataImplTest, MissingOrEmptyModelSpec) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  request.add_metadata_field(kSignatureDef);
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                   &response)
                .code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                   &response)
                .code());

  // Model name is wrong, not found.
  model_spec->set_name("test");
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                   &response)
                .code());
}

TEST_P(GetModelMetadataImplTest, ReturnsSignaturesForValidModel) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  request.add_metadata_field(kSignatureDef);

  TF_EXPECT_OK(GetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                      &response));
  EXPECT_THAT(response.model_spec(),
              test_util::EqualsProto(request.model_spec()));
  EXPECT_EQ(response.metadata_size(), 1);
  SignatureDefMap received_signature_def_map;
  response.metadata().at(kSignatureDef).UnpackTo(&received_signature_def_map);

  SignatureDefMap expected_signature_def_map =
      GetSignatureDefMap(GetServerCore(), request.model_spec());
  EXPECT_THAT(response.model_spec(),
              test_util::EqualsProto(request.model_spec()));

  EXPECT_EQ(expected_signature_def_map.signature_def().size(),
            received_signature_def_map.signature_def().size());
  if (GetParam()) {
    EXPECT_THAT(
        expected_signature_def_map.signature_def().at("regress_x_to_y"),
        test_util::EqualsProto(
            received_signature_def_map.signature_def().at("regress_x_to_y")));
  } else {
    EXPECT_THAT(expected_signature_def_map.signature_def().at("regress"),
                test_util::EqualsProto(
                    received_signature_def_map.signature_def().at("regress")));
  }
  EXPECT_THAT(
      expected_signature_def_map.signature_def().at(
          kDefaultServingSignatureDefKey),
      test_util::EqualsProto(received_signature_def_map.signature_def().at(
          kDefaultServingSignatureDefKey)));
}

// Test all ClassifierTest test cases with both SessionBundle and SavedModel.
INSTANTIATE_TEST_CASE_P(UseSavedModel, GetModelMetadataImplTest,
                        ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
