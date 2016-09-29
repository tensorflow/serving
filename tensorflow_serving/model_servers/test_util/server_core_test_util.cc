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

#include "tensorflow_serving/model_servers/test_util/server_core_test_util.h"

#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

std::vector<ServableId> ServerCoreTestAccess::ListAvailableServableIds() const {
  return core_->ListAvailableServableIds();
}

ModelServerConfig ServerCoreTest::GetTestModelServerConfig() {
  ModelServerConfig config;
  auto model = config.mutable_model_config_list()->add_config();
  model->set_name(kTestModelName);
  model->set_base_path(test_util::TestSrcDirPath(
      "/servables/tensorflow/testdata/half_plus_two"));
  model->set_model_platform(kTensorFlowModelPlatform);
  return config;
}

ServerCoreConfig ServerCoreTest::GetTestServerCoreConfig() {
  ServerCoreConfig config;
  config.file_system_poll_wait_seconds = 0;
  return config;
}

Status ServerCoreTest::CreateServerCore(
    const ModelServerConfig& config, std::unique_ptr<ServerCore>* server_core) {
  return CreateServerCore(
      config, test_util::FakeLoaderSourceAdapter::GetCreator(), server_core);
}

Status ServerCoreTest::CreateServerCore(
    const ModelServerConfig& config,
    const ServerCore::SourceAdapterCreator& source_adapter_creator,
    std::unique_ptr<ServerCore>* server_core) {
  return ServerCore::Create(
      config, source_adapter_creator,  // ServerCore::SourceAdapterCreator
      [](EventBus<ServableState>* event_bus,
         std::unique_ptr<ServableStateMonitor>* monitor) -> Status {
        monitor->reset(new ServableStateMonitor(event_bus));
        return Status::OK();
      },  // ServerCore::ServableStateMonitor
      [](const ::google::protobuf::Any& any, EventBus<ServableState>* event_bus,
         Target<std::unique_ptr<Loader>>* target) -> Status {
        return Status::OK();
      },  // ServerCore::CustomModelConfigLoader
      GetTestServerCoreConfig(),
      server_core);
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
