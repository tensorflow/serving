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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"
#include "tensorflow_serving/model_servers/test_util/server_core_test_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::Eq;

tensorflow::Status CreateSourceAdapter(
    const string& model_type,
    std::unique_ptr<ServerCore::ModelServerSourceAdapter>* adapter) {
  adapter->reset(new test_util::FakeLoaderSourceAdapter);
  return Status::OK();
}

tensorflow::Status CreateServableStateMonitor(
    EventBus<ServableState>* event_bus,
    std::unique_ptr<ServableStateMonitor>* monitor) {
  monitor->reset(new ServableStateMonitor(event_bus));
  return tensorflow::Status::OK();
}

tensorflow::Status LoadDynamicModelConfig(
    const ::google::protobuf::Any& any,
    Target<std::unique_ptr<Loader>>* target) {
  CHECK(false);
}

ModelServerConfig CreateModelServerConfig() {
  ModelServerConfig config;
  auto model = config.mutable_model_config_list()->add_config();
  model->set_name("test_model");
  model->set_base_path(test_util::TestSrcDirPath(
      "/servables/tensorflow/testdata/half_plus_two"));
  model->set_model_type("tensorflow");
  return config;
}

// TODO(b/29012372): Currently we only support a single config reload.
// Verify multiple calls result in an error.
TEST(ServerCoreTest, MultipleLoadConfigs) {
  // Create with an empty config.
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(ModelServerConfig(), &CreateSourceAdapter,
                                  &CreateServableStateMonitor,
                                  &LoadDynamicModelConfig, &server_core));
  // Reload with a populated config.  This is allowed since the previous config
  // was empty.
  TF_ASSERT_OK(server_core->ReloadConfig(CreateModelServerConfig()));

  // Second reload fails since the previous config has models.
  EXPECT_THAT(
      server_core->ReloadConfig({}).ToString(),
      ::testing::HasSubstr("Repeated ReloadConfig calls not supported"));
}

TEST(ServerCoreTest, CreateWaitsTillModelsAvailable) {
  std::unique_ptr<ServerCore> server_core;
  TF_ASSERT_OK(ServerCore::Create(
      CreateModelServerConfig(), &CreateSourceAdapter,
      &CreateServableStateMonitor, &LoadDynamicModelConfig, &server_core));

  const std::vector<ServableId> available_servables =
      test_util::ServerCoreTestAccess(server_core.get())
          .ListAvailableServableIds();
  ASSERT_EQ(available_servables.size(), 1);
  const ServableId expected_id = {"test_model", 123};
  EXPECT_EQ(available_servables.at(0), expected_id);
}

TEST(ServerCoreTest, ErroringModel) {
  std::unique_ptr<ServerCore> server_core;
  const Status status = ServerCore::Create(
      CreateModelServerConfig(),
      [](const string& model_type,
         std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
             source_adapter) -> Status {
        source_adapter->reset(
            new ErrorInjectingSourceAdapter<StoragePath,
                                            std::unique_ptr<Loader>>(
                Status(error::CANCELLED, "")));
        return Status::OK();
      },
      &CreateServableStateMonitor, &LoadDynamicModelConfig, &server_core);
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
