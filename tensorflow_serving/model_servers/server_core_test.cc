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

#include "tensorflow_serving/model_servers/server_core.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/test_util/server_core_test_util.h"
#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.h"
#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.proto.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::ServerCoreTest;

TEST_P(ServerCoreTest, CreateWaitsTillModelsAvailable) {
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(CreateServerCore(GetTestModelServerConfigForFakePlatform(),
                                &server_core));

  const std::vector<ServableId> available_servables =
      server_core->ListAvailableServableIds();
  ASSERT_EQ(available_servables.size(), 1);
  const ServableId expected_id = {test_util::kTestModelName,
                                  test_util::kTestModelVersion};
  EXPECT_EQ(available_servables.at(0), expected_id);

  ModelSpec model_spec;
  model_spec.set_name(test_util::kTestModelName);
  model_spec.mutable_version()->set_value(test_util::kTestModelVersion);
  ServableHandle<string> servable_handle;
  TF_ASSERT_OK(
      server_core->GetServableHandle<string>(model_spec, &servable_handle));
  EXPECT_EQ(servable_handle.id(), expected_id);
}

TEST_P(ServerCoreTest, ReloadConfigWaitsTillModelsAvailable) {
  // Create a server with no models, initially.
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(CreateServerCore(ModelServerConfig(), &server_core));

  // Reconfigure it to load our test model.
  TF_ASSERT_OK(
      server_core->ReloadConfig(GetTestModelServerConfigForFakePlatform()));

  const std::vector<ServableId> available_servables =
      server_core->ListAvailableServableIds();
  ASSERT_EQ(available_servables.size(), 1);
  const ServableId expected_id = {test_util::kTestModelName,
                                  test_util::kTestModelVersion};
  EXPECT_EQ(available_servables.at(0), expected_id);
}

TEST_P(ServerCoreTest, ErroringModel) {
  ServerCore::Options options = GetDefaultOptions();
  test_util::StoragePathErrorInjectingSourceAdapterConfig source_adapter_config;
  source_adapter_config.set_error_message("injected error");
  ::google::protobuf::Any source_adapter_config_any;
  source_adapter_config_any.PackFrom(source_adapter_config);
  (*(*options.platform_config_map.mutable_platform_configs())[kFakePlatform]
        .mutable_source_adapter_config()) = source_adapter_config_any;
  options.model_server_config = GetTestModelServerConfigForFakePlatform();
  std::unique_ptr<ServerCore> server_core;
  Status status = ServerCore::Create(std::move(options), &server_core);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Some models did not become available"));
}

TEST_P(ServerCoreTest, IllegalReconfigurationToCustomConfig) {
  // Create a ServerCore with ModelConfigList config.
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(CreateServerCore(GetTestModelServerConfigForFakePlatform(),
                                &server_core));

  // Reload with a custom config. This is not allowed since the server was
  // first configured with TensorFlow model platform.
  ModelServerConfig config;
  config.mutable_custom_model_config();
  EXPECT_THAT(server_core->ReloadConfig(config).ToString(),
              ::testing::HasSubstr("Cannot transition to requested config"));
}

TEST_P(ServerCoreTest, IllegalReconfigurationFromCustomConfig) {
  // Create a ServerCore with custom config.
  std::unique_ptr<ServerCore> server_core;
  ModelServerConfig config;
  config.mutable_custom_model_config();
  TF_ASSERT_OK(CreateServerCore(config, &server_core));

  // Reload with a ModelConfigList config. This is not allowed, since the
  // server was first configured with a custom config.
  EXPECT_THAT(
      server_core->ReloadConfig(GetTestModelServerConfigForFakePlatform())
          .ToString(),
      ::testing::HasSubstr("Cannot transition to requested config"));
}

TEST_P(ServerCoreTest, IllegalConfigModelTypeAndPlatformSet) {
  // Create a ServerCore with both model_type and model_platform set.
  std::unique_ptr<ServerCore> server_core;
  ModelServerConfig config = GetTestModelServerConfigForFakePlatform();
  config.mutable_model_config_list()->mutable_config(0)->set_model_type(
      ModelType::TENSORFLOW);
  EXPECT_THAT(CreateServerCore(config, &server_core).ToString(),
              ::testing::HasSubstr("Illegal setting both"));
}

TEST_P(ServerCoreTest, DeprecatedModelTypeConfig) {
  // Create a ServerCore with deprecated config.
  std::unique_ptr<ServerCore> server_core;
  ModelServerConfig config = GetTestModelServerConfigForTensorflowPlatform();
  config.mutable_model_config_list()->mutable_config(0)->set_model_platform("");
  config.mutable_model_config_list()->mutable_config(0)->set_model_type(
      ModelType::TENSORFLOW);
  TF_ASSERT_OK(CreateServerCore(config, &server_core));

  const std::vector<ServableId> available_servables =
      server_core->ListAvailableServableIds();
  ASSERT_EQ(available_servables.size(), 1);
  const ServableId expected_id = {test_util::kTestModelName,
                                  test_util::kTestModelVersion};
  EXPECT_EQ(available_servables.at(0), expected_id);
}

INSTANTIATE_TEST_CASE_P(
    TestType, ServerCoreTest,
    ::testing::Range(0, static_cast<int>(ServerCoreTest::NUM_TEST_TYPES)));

}  // namespace
}  // namespace serving
}  // namespace tensorflow
