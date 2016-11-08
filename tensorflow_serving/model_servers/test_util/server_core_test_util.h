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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_SERVER_CORE_TEST_UTIL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_SERVER_CORE_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {
namespace test_util {

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;

class ServerCoreTest : public ::testing::Test {
 protected:
  // Returns ModelServerConfig that contains test model.
  ModelServerConfig GetTestModelServerConfig();

  // Create a ServerCore object configured to use FakeLoaderSourceAdapter.
  Status CreateServerCore(const ModelServerConfig& config,
                          std::unique_ptr<ServerCore>* server_core);

  // Create a ServerCore object with the supplied SourceAdapterCreator.
  Status CreateServerCore(
      const ModelServerConfig& config,
      const ServerCore::SourceAdapterCreator& source_adapter_creator,
      std::unique_ptr<ServerCore>* server_core);

  // Create a ServerCore object with the supplied options. The ServerCore uses
  // continuous polling to speed up testing.
  Status CreateServerCore(ServerCore::Options options,
                          std::unique_ptr<ServerCore>* server_core);
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_SERVER_CORE_TEST_UTIL_H_
