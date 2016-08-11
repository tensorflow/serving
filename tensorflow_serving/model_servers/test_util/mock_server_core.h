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

#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/apis/model.proto.h"
#include "tensorflow_serving/config/model_server_config.proto.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockServerCore : public ServerCore {
 public:
  explicit MockServerCore(SourceAdapterCreator source_adapter_creator)
      : ServerCore(source_adapter_creator) {}

  MOCK_CONST_METHOD0(servable_state_monitor, ServableStateMonitor*());
  MOCK_METHOD1(ReloadConfig, Status(const ModelServerConfig&));

  template <typename T>
  Status GetServableHandle(const ModelSpec& model_spec,
                           ServableHandle<T>* const handle) {
    LOG(FATAL) << "Not implemented.";
  }
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_MOCK_SERVER_CORE_H_
