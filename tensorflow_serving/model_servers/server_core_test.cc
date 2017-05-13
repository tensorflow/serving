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

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.pb.h"
#include "tensorflow_serving/core/test_util/fake_log_collector.h"
#include "tensorflow_serving/core/test_util/mock_request_logger.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/test_util/server_core_test_util.h"
#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.h"
#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::Invoke;
using ::testing::NiceMock;
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

TEST_P(ServerCoreTest, ReloadConfigUnloadsModels) {
  const ModelServerConfig nonempty_config =
      GetTestModelServerConfigForFakePlatform();
  ModelServerConfig empty_config;
  empty_config.mutable_model_config_list();

  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(CreateServerCore(nonempty_config, &server_core));
  ASSERT_FALSE(server_core->ListAvailableServableIds().empty());

  TF_ASSERT_OK(server_core->ReloadConfig(empty_config));
  // Wait for the unload to finish (ReloadConfig() doesn't block on this).
  while (!server_core->ListAvailableServableIds().empty()) {
    Env::Default()->SleepForMicroseconds(10 * 1000);
  }
}

TEST_P(ServerCoreTest, ReloadConfigHandlesLoadingAPreviouslyUnloadedModel) {
  ModelServerConfig empty_config;
  empty_config.mutable_model_config_list();
  const ModelServerConfig nonempty_config =
      GetTestModelServerConfigForFakePlatform();

  // Load, and then unload, a servable.
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(CreateServerCore(nonempty_config, &server_core));
  TF_ASSERT_OK(server_core->ReloadConfig(empty_config));
  // Wait for the unload to finish (ReloadConfig() doesn't block on this).
  while (!server_core->ListAvailableServableIds().empty()) {
    Env::Default()->SleepForMicroseconds(10 * 1000);
  }

  // Re-load the same servable.
  TF_ASSERT_OK(server_core->ReloadConfig(nonempty_config));
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
  (*(*options.platform_config_map
          .mutable_platform_configs())[test_util::kFakePlatform]
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

TEST_P(ServerCoreTest, DuplicateModelNameInConfig) {
  std::unique_ptr<ServerCore> server_core;
  ModelServerConfig config = GetTestModelServerConfigForTensorflowPlatform();
  *config.mutable_model_config_list()->add_config() =
      config.model_config_list().config(0);
  EXPECT_FALSE(CreateServerCore(config, &server_core).ok());
}

TEST_P(ServerCoreTest, UnknownModelPlatform) {
  std::unique_ptr<ServerCore> server_core;
  ModelServerConfig config = GetTestModelServerConfigForTensorflowPlatform();
  config.mutable_model_config_list()->mutable_config(0)->set_model_platform(
      "not_a_known_platform");
  EXPECT_FALSE(CreateServerCore(config, &server_core).ok());
}

// Creates a model name that incorporates 'platform'. Useful for tests that have
// one model for a given platform.
string ModelNameForPlatform(const string& platform) {
  return strings::StrCat("model_for_", platform);
}

// Builds a ModelSpec with a model named 'ModelNameForPlatform(platform)' and
// version 0.
ModelSpec ModelSpecForPlatform(const string& platform) {
  ModelSpec spec;
  spec.set_name(ModelNameForPlatform(platform));
  spec.mutable_version()->set_value(0);
  return spec;
}

// Builds a ModelConfig with a model named 'ModelNameForPlatform(platform)',
// base path '<root_path>/<model_name>' and platform 'platform'.
ModelConfig ModelConfigForPlatform(const string& root_path,
                                   const string& platform) {
  const string model_name = ModelNameForPlatform(platform);
  ModelConfig config;
  config.set_name(model_name);
  config.set_base_path(io::JoinPath(root_path, model_name));
  config.set_model_platform(platform);
  return config;
}

// Creates a directory for the given version of the model.
void CreateModelDir(const ModelConfig& model_config, int version) {
  TF_CHECK_OK(Env::Default()->CreateDir(model_config.base_path()));
  const string version_str = strings::StrCat(version);
  TF_CHECK_OK(Env::Default()->CreateDir(
      io::JoinPath(model_config.base_path(), version_str)));
}

// Adds 'platform' to 'platform_config_map' with a fake source adapter that
// uses suffix 'suffix_for_<platform>'.
void CreateFakePlatform(const string& platform,
                        PlatformConfigMap* platform_config_map) {
  test_util::FakeLoaderSourceAdapterConfig source_adapter_config;
  source_adapter_config.set_suffix(strings::StrCat("suffix_for_", platform));
  ::google::protobuf::Any source_adapter_config_any;
  source_adapter_config_any.PackFrom(source_adapter_config);
  (*(*platform_config_map->mutable_platform_configs())[platform]
        .mutable_source_adapter_config()) = source_adapter_config_any;
}

// Constructs the servable data that a platform's fake source adapter will emit.
string ServableDataForPlatform(const string& root_path, const string& platform,
                               int version) {
  const string version_str = strings::StrCat(version);
  return io::JoinPath(root_path, ModelNameForPlatform(platform), version_str,
                      strings::StrCat("suffix_for_", platform));
}

TEST_P(ServerCoreTest, MultiplePlatforms) {
  const string root_path = io::JoinPath(
      testing::TmpDir(), strings::StrCat("MultiplePlatforms_", GetTestType()));
  TF_ASSERT_OK(Env::Default()->CreateDir(root_path));

  // Create a ServerCore with two platforms, and one model for each platform.
  ServerCore::Options options = GetDefaultOptions();
  options.platform_config_map.Clear();
  const std::vector<string> platforms = {"platform_0", "platform_1"};
  for (const string& platform : platforms) {
    CreateFakePlatform(platform, &options.platform_config_map);
    const ModelConfig model_config =
        ModelConfigForPlatform(root_path, platform);
    *options.model_server_config.mutable_model_config_list()->add_config() =
        model_config;
    CreateModelDir(model_config, 0 /* version */);
  }
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core));

  // Verify the models got loaded via the platform-specific source adapters.
  for (const string& platform : platforms) {
    ServableHandle<string> servable_handle;
    TF_ASSERT_OK(server_core->GetServableHandle<string>(
        ModelSpecForPlatform(platform), &servable_handle));
    const string model_name = ModelNameForPlatform(platform);
    const auto expected_servable_id = ServableId{model_name, 0};
    EXPECT_EQ(servable_handle.id(), expected_servable_id);
    EXPECT_EQ(ServableDataForPlatform(root_path, platform, 0 /* version */),
              *servable_handle);
  }
}

TEST_P(ServerCoreTest, MultiplePlatformsWithConfigChange) {
  const string root_path = io::JoinPath(
      testing::TmpDir(),
      strings::StrCat("MultiplePlatformsWithConfigChange_", GetTestType()));
  TF_ASSERT_OK(Env::Default()->CreateDir(root_path));

  // Create config for three platforms, and one model per platform.
  ServerCore::Options options = GetDefaultOptions();
  options.platform_config_map.Clear();
  const std::vector<string> platforms = {"platform_0", "platform_1",
                                         "platform_2"};
  std::vector<ModelConfig> models;
  for (const string& platform : platforms) {
    CreateFakePlatform(platform, &options.platform_config_map);
    const ModelConfig model_config =
        ModelConfigForPlatform(root_path, platform);
    models.push_back(model_config);
    CreateModelDir(model_config, 0 /* version */);
  }

  auto verify_model_loaded = [&root_path](ServerCore* server_core,
                                          const string& platform) {
    ServableHandle<string> servable_handle;
    TF_ASSERT_OK(server_core->GetServableHandle<string>(
        ModelSpecForPlatform(platform), &servable_handle));
    const string model_name = ModelNameForPlatform(platform);
    const auto expected_servable_id = ServableId{model_name, 0};
    EXPECT_EQ(servable_handle.id(), expected_servable_id);
    EXPECT_EQ(ServableDataForPlatform(root_path, platform, 0 /* version */),
              *servable_handle);
  };
  auto verify_model_not_loaded = [&root_path](ServerCore* server_core,
                                              const string& platform) {
    ServableHandle<string> servable_handle;
    EXPECT_FALSE(server_core
                     ->GetServableHandle<string>(ModelSpecForPlatform(platform),
                                                 &servable_handle)
                     .ok());
  };

  // Initially configure the ServerCore to have models 0 and 1.
  ModelServerConfig* initial_model_config = &options.model_server_config;
  (*initial_model_config->mutable_model_config_list()->add_config()) =
      models[0];
  (*initial_model_config->mutable_model_config_list()->add_config()) =
      models[1];
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core));
  verify_model_loaded(server_core.get(), platforms[0]);
  verify_model_loaded(server_core.get(), platforms[1]);
  verify_model_not_loaded(server_core.get(), platforms[2]);

  // Reload with models 1 and 2.
  ModelServerConfig new_model_config;
  (*new_model_config.mutable_model_config_list()->add_config()) = models[1];
  (*new_model_config.mutable_model_config_list()->add_config()) = models[2];
  TF_ASSERT_OK(server_core->ReloadConfig(new_model_config));
  verify_model_not_loaded(server_core.get(), platforms[0]);
  verify_model_loaded(server_core.get(), platforms[1]);
  verify_model_loaded(server_core.get(), platforms[2]);
}

TEST_P(ServerCoreTest, IllegalToChangeModelPlatform) {
  const string root_path = io::JoinPath(
      testing::TmpDir(),
      strings::StrCat("IllegalToChangeModelPlatform_", GetTestType()));
  TF_ASSERT_OK(Env::Default()->CreateDir(root_path));

  ServerCore::Options options = GetDefaultOptions();
  options.platform_config_map.Clear();
  const std::vector<string> platforms = {"platform_0", "platform_1"};
  for (const string& platform : platforms) {
    CreateFakePlatform(platform, &options.platform_config_map);
  }

  // Configure a model for platform 0.
  ModelServerConfig initial_config;
  const ModelConfig model_config =
      ModelConfigForPlatform(root_path, platforms[0]);
  *initial_config.mutable_model_config_list()->add_config() = model_config;
  CreateModelDir(model_config, 0 /* version */);

  options.model_server_config = initial_config;
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core));

  // Attempt to switch the existing model to platform 1.
  ModelServerConfig new_config = initial_config;
  new_config.mutable_model_config_list()->mutable_config(0)->set_model_platform(
      platforms[1]);
  const Status reconfigure_status = server_core->ReloadConfig(new_config);
  EXPECT_FALSE(reconfigure_status.ok());
  EXPECT_THAT(reconfigure_status.ToString(),
              ::testing::HasSubstr("Illegal to change a model's platform"));
}

TEST_P(ServerCoreTest, RequestLoggingOff) {
  // Create a ServerCore with deprecated config.
  std::unique_ptr<ServerCore> server_core;
  const ModelServerConfig config =
      GetTestModelServerConfigForTensorflowPlatform();
  TF_ASSERT_OK(CreateServerCore(config, &server_core));

  TF_ASSERT_OK(
      server_core->Log(PredictRequest(), PredictResponse(), LogMetadata()));
}

TEST_P(ServerCoreTest, RequestLoggingOn) {
  std::unordered_map<string, FakeLogCollector*> log_collector_map;
  ServerCore::Options options = GetDefaultOptions();
  TF_CHECK_OK(ServerRequestLogger::Create(
      [&](const LoggingConfig& logging_config,
          std::unique_ptr<RequestLogger>* const request_logger) {
        const string& filename_prefix =
            logging_config.log_collector_config().filename_prefix();
        log_collector_map[filename_prefix] = new FakeLogCollector();
        auto mock_request_logger = std::unique_ptr<NiceMock<MockRequestLogger>>(
            new NiceMock<MockRequestLogger>(
                logging_config, log_collector_map[filename_prefix]));
        ON_CALL(*mock_request_logger, CreateLogMessage(_, _, _, _))
            .WillByDefault(Invoke([&](const google::protobuf::Message& actual_request,
                                      const google::protobuf::Message& actual_response,
                                      const LogMetadata& actual_log_metadata,
                                      std::unique_ptr<google::protobuf::Message>* log) {
              *log = std::unique_ptr<google::protobuf::Any>(
                  new google::protobuf::Any());
              return Status::OK();
            }));
        *request_logger = std::move(mock_request_logger);
        return Status::OK();
      },
      &options.server_request_logger));

  // We now setup a model-server-config with a model which switches on request
  // logging.
  LogCollectorConfig log_collector_config;
  log_collector_config.set_type("");
  log_collector_config.set_filename_prefix(test_util::kTestModelName);
  LoggingConfig logging_config;
  *logging_config.mutable_log_collector_config() = log_collector_config;
  logging_config.mutable_sampling_config()->set_sampling_rate(1.0);

  ModelServerConfig model_server_config =
      GetTestModelServerConfigForTensorflowPlatform();
  *model_server_config.mutable_model_config_list()
       ->mutable_config(0)
       ->mutable_logging_config() = logging_config;
  options.model_server_config = model_server_config;

  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(std::move(options), &server_core));

  LogMetadata log_metadata0;
  auto* const model_spec0 = log_metadata0.mutable_model_spec();
  model_spec0->set_name(test_util::kTestModelName);
  TF_ASSERT_OK(
      server_core->Log(PredictRequest(), PredictResponse(), log_metadata0));
  ASSERT_EQ(1, log_collector_map.size());
  EXPECT_EQ(1, log_collector_map[test_util::kTestModelName]->collect_count());
}

INSTANTIATE_TEST_CASE_P(
    TestType, ServerCoreTest,
    ::testing::Range(0, static_cast<int>(ServerCoreTest::NUM_TEST_TYPES)));

}  // namespace
}  // namespace serving
}  // namespace tensorflow
