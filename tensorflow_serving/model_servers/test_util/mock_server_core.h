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

// GMock implementation of ServerCore.
#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_MOCK_SERVER_CORE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_MOCK_SERVER_CORE_H_

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/map.h"
#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/apis/logging.proto.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/server_request_logger.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.pb.h"
#include "tensorflow_serving/core/test_util/servable_handle_test_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockServerCore : public ServerCore {
 public:
  // Creates a PlatformConfigMap with a FakeLoaderSourceAdapterConfig.
  static PlatformConfigMap CreateFakeLoaderPlatformConfigMap() {
    ::google::protobuf::Any source_adapter_config;
    source_adapter_config.PackFrom(test_util::FakeLoaderSourceAdapterConfig());
    PlatformConfigMap platform_config_map;
    (*(*platform_config_map.mutable_platform_configs())["fake_servable"]
          .mutable_source_adapter_config()) = source_adapter_config;
    return platform_config_map;
  }

  static Options GetOptions(
      const PlatformConfigMap& platform_config_map,
      std::unique_ptr<ServerRequestLogger> server_request_logger) {
    Options options;
    options.platform_config_map = platform_config_map;
    options.servable_state_monitor_creator =
        [](EventBus<ServableState>* event_bus,
           std::unique_ptr<ServableStateMonitor>* monitor) -> Status {
      monitor->reset(new ServableStateMonitor(event_bus));
      return Status();
    };
    options.custom_model_config_loader =
        [](const ::google::protobuf::Any& any,
           EventBus<ServableState>* event_bus,
           UniquePtrWithDeps<AspiredVersionsManager>* manager) -> Status {
      return Status();
    };
    if (server_request_logger != nullptr) {
      options.server_request_logger = std::move(server_request_logger);
    } else {
      TF_CHECK_OK(
          ServerRequestLogger::Create(nullptr, &options.server_request_logger));
    }
    return options;
  }

  explicit MockServerCore(const PlatformConfigMap& platform_config_map)
      : MockServerCore(platform_config_map, nullptr) {}

  MockServerCore(const PlatformConfigMap& platform_config_map,
                 std::unique_ptr<ServerRequestLogger> server_request_logger)
      : ServerCore(GetOptions(platform_config_map,
                              std::move(server_request_logger))) {}

  MOCK_METHOD(ServableStateMonitor*, servable_state_monitor, (),
              (const, override));
  MOCK_METHOD(Status, ReloadConfig, (const ModelServerConfig&), (override));
  MOCK_METHOD(Status, Log,
              (const google::protobuf::Message& request, const google::protobuf::Message& response,
               const LogMetadata& log_metadata),
              (override));

  // Sets the Servable used by GetServableHandle
  void SetServable(std::unique_ptr<Servable> servable) {
    servable_ = std::move(servable);
  }

  template <typename T>
  Status GetServableHandle(const ModelSpec& model_spec,
                           ServableHandle<T>* const handle) {
    LOG(FATAL) << "Not implemented.";
  }

  // Implement GetServable for type Servable.  Will return the Servable
  // set by SetServable, otherwise forwards to base class.
  virtual Status GetServableHandle(
      const ModelSpec& model_spec,
      ServableHandle<Servable>* const handle) override {
    if (servable_) {
      const ServableId id = {"servable", 0};
      *handle = WrapAsHandle<Servable>(id, servable_.get());
      return absl::OkStatus();
    } else {
      return ServerCore::GetServableHandle<Servable>(model_spec, handle);
    }
  }

  std::unique_ptr<Servable> servable_;
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_MOCK_SERVER_CORE_H_
