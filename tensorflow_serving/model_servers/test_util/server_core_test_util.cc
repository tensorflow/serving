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

#include "tensorflow_serving/core/eager_load_policy.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

ModelServerConfig ServerCoreTest::GetTestModelServerConfig() {
  ModelServerConfig config;
  auto model = config.mutable_model_config_list()->add_config();
  model->set_name(kTestModelName);
  model->set_base_path(test_util::TestSrcDirPath(
      "/servables/tensorflow/testdata/half_plus_two"));
  model->set_model_platform(kTensorFlowModelPlatform);
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
  // For ServerCore Options, we leave servable_state_monitor_creator unspecified
  // so the default servable_state_monitor_creator will be used.
  ServerCore::Options options;
  options.model_server_config = config;
  options.source_adapter_creator = source_adapter_creator;
  options.custom_model_config_loader = [](
      const ::google::protobuf::Any& any, EventBus<ServableState>* event_bus,
      UniquePtrWithDeps<AspiredVersionsManager>* manager) -> Status {
    return Status::OK();
  };
  return CreateServerCore(std::move(options), server_core);
}

Status ServerCoreTest::CreateServerCore(
    ServerCore::Options options, std::unique_ptr<ServerCore>* server_core) {
  options.file_system_poll_wait_seconds = 0;
  if (options.aspired_version_policy == nullptr) {
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new EagerLoadPolicy);
  }
  return ServerCore::Create(std::move(options), server_core);
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
