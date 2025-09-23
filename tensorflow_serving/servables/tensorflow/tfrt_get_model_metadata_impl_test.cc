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

#include "tensorflow_serving/servables/tensorflow/tfrt_get_model_metadata_impl.h"

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/wrappers.pb.h"
#include "google/protobuf/map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tsl/platform/casts.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;
constexpr absl::string_view kSignatureDef = "signature_def";

class TFRTGetModelMetadataImplTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    tfrt_stub::SetGlobalRuntime(
        tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));

    const string saved_model_path = test_util::TensorflowTestSrcDirPath(
        "cc/saved_model/testdata/half_plus_two");
    TF_ASSERT_OK(
        CreateServerCore(saved_model_path, true, &saved_model_server_core_));
  }

  static void TearDownTestSuite() { saved_model_server_core_.reset(); }

 protected:
  static absl::Status CreateServerCore(
      const string& model_path, bool saved_model_on_disk,
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
    return ServerCore::Create(std::move(options), server_core);
  }

  ServerCore* GetServerCore() { return saved_model_server_core_.get(); }

 private:
  static std::unique_ptr<ServerCore> saved_model_server_core_;
};

std::unique_ptr<ServerCore>
    TFRTGetModelMetadataImplTest::saved_model_server_core_;

SignatureDefMap GetSignatureDefMap(ServerCore* server_core,
                                   const ModelSpec& model_spec) {
  SignatureDefMap signature_def_map;
  ServableHandle<Servable> servable;
  TF_EXPECT_OK(server_core->GetServableHandle(model_spec, &servable));
  auto& saved_model =
      down_cast<TfrtSavedModelServable*>(servable.get())->saved_model();
  for (const auto& signature : saved_model.GetMetaGraphDef().signature_def()) {
    (*signature_def_map.mutable_signature_def())[signature.first] =
        signature.second;
  }
  return signature_def_map;
}

TEST_F(TFRTGetModelMetadataImplTest, EmptyOrInvalidMetadataFieldList) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  // Empty metadata field list is invalid.
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());
  request.add_metadata_field("some_stuff");

  // Field enum is outside of valid range.
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());
}

TEST_F(TFRTGetModelMetadataImplTest, MissingOrEmptyModelSpec) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  request.add_metadata_field(std::string(kSignatureDef));
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->clear_name();

  // Model name is not specified.
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());

  // Model name is wrong, not found.
  model_spec->set_name("test");
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());
}

TEST_F(TFRTGetModelMetadataImplTest, ReturnsSignaturesForValidModel) {
  GetModelMetadataRequest request;
  GetModelMetadataResponse response;

  ModelSpec* model_spec = request.mutable_model_spec();
  model_spec->set_name(kTestModelName);
  model_spec->mutable_version()->set_value(kTestModelVersion);
  request.add_metadata_field(std::string(kSignatureDef));

  TF_EXPECT_OK(TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(),
                                                          request, &response));
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
  EXPECT_THAT(
      expected_signature_def_map.signature_def().at("regress_x_to_y"),
      test_util::EqualsProto(
          received_signature_def_map.signature_def().at("regress_x_to_y")));
  EXPECT_THAT(
      expected_signature_def_map.signature_def().at(
          kDefaultServingSignatureDefKey),
      test_util::EqualsProto(received_signature_def_map.signature_def().at(
          kDefaultServingSignatureDefKey)));
}

// Verifies that GetModelMetadataWithModelSpec() uses the model spec override
// rather than the one in the request.
TEST_F(TFRTGetModelMetadataImplTest, ModelSpecOverride) {
  auto request = test_util::CreateProto<GetModelMetadataRequest>(
      "model_spec {"
      "  name: \"test_model\""
      "}");
  request.add_metadata_field(std::string(kSignatureDef));
  auto model_spec_override =
      test_util::CreateProto<ModelSpec>("name: \"nonexistent_model\"");

  GetModelMetadataResponse response;
  EXPECT_NE(tensorflow::error::NOT_FOUND,
            TFRTGetModelMetadataImpl::GetModelMetadata(GetServerCore(), request,
                                                       &response)
                .code());
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            TFRTGetModelMetadataImpl::GetModelMetadataWithModelSpec(
                GetServerCore(), model_spec_override, request, &response)
                .code());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
