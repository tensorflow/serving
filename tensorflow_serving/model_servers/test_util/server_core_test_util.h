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

#include <array>
#include <utility>

#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A test utility that provides access to private ServerCore members.
class ServerCoreTestAccess {
 public:
  explicit ServerCoreTestAccess(ServerCore* core) : core_(core) {}

  ServerCore::Options* mutable_options() { return &core_->options_; }

 private:
  ServerCore* const core_;
};

constexpr char kTestModelName[] = "test_model";
constexpr int kTestModelVersion = 123;
constexpr int kTestModelLargerVersion = 124;
constexpr int kTestModelBogusVersion = 777;
constexpr std::array<int64_t, 2> kAspiredVersions = {kTestModelVersion,
                                                     kTestModelLargerVersion};
// The name of the platform associated with FakeLoaderSourceAdapter.
constexpr char kFakePlatform[] = "fake_servable";

// ServerCoreTest is parameterized based on the TestType enum defined below.
// TODO(b/32248363): remove the parameter and TestType after we switch Model
// Server to Saved Model.
class ServerCoreTest : public ::testing::TestWithParam<std::tuple<int, bool>> {
 public:
  // The parameter of this test.
  enum TestType {
    // SavedModelBundle is used on export.
    SAVED_MODEL_BACKWARD_COMPATIBILITY,
    // SavedModelBundle is used on native Saved Model.
    SAVED_MODEL,
    // This should always be the last value.
    NUM_TEST_TYPES,
  };

  static string GetNameOfTestType(int test_type) {
    switch (static_cast<TestType>(test_type)) {
      case SAVED_MODEL_BACKWARD_COMPATIBILITY:
        return "SAVED_MODEL_BACKWARD_COMPATIBILITY";
      case SAVED_MODEL:
        return "SAVED_MODEL";
      default:
        return "unknown";
    }
  }

  // Creates some reasonable default ServerCore options for tests.
  static ServerCore::Options GetDefaultOptions();

 protected:
  // Returns ModelServerConfig that contains test model for the fake platform.
  ModelServerConfig GetTestModelServerConfigForFakePlatform();

  // Returns ModelServerConfig that contains test model for the tensorflow
  // platform.
  ModelServerConfig GetTestModelServerConfigForTensorflowPlatform();

  // Mutates 'config' by changing the model's base path to point to a variant
  // of half-plus-two that has two versions instead of one.
  void SwitchToHalfPlusTwoWith2Versions(ModelServerConfig* config);

  // Creates a ServerCore object configured with both a fake platform and the
  // tensorflow platform, using the supplied options.
  Status CreateServerCore(const ModelServerConfig& config,
                          ServerCore::Options options,
                          std::unique_ptr<ServerCore>* server_core);

  // Creates a ServerCore object configured with both a fake platform and the
  // tensorflow platform, using GetDefaultOptions().
  Status CreateServerCore(const ModelServerConfig& config,
                          std::unique_ptr<ServerCore>* server_core) {
    return CreateServerCore(config, GetDefaultOptions(), server_core);
  }

  // Returns test type.
  // This is the first parameter of this test.
  TestType GetTestType() {
    return static_cast<TestType>(std::get<0>(GetParam()));
  }

  // Returns whether to assume paths are URIs.
  // This is the second parameter of this test.
  bool PrefixPathsWithURIScheme() { return std::get<1>(GetParam()); }

  // Returns a string corresponding to the parameterized test-case.
  string GetNameForTestCase() {
    return GetNameOfTestType(GetTestType()) + "_" +
           (PrefixPathsWithURIScheme() ? "URI" : "Path");
  }
};

// Creates a ServerCore object with the supplied options.
Status CreateServerCore(const ModelServerConfig& config,
                        ServerCore::Options options,
                        std::unique_ptr<ServerCore>* server_core);

// Creates a ServerCore object with sane defaults.
Status CreateServerCore(const ModelServerConfig& config,
                        std::unique_ptr<ServerCore>* server_core);

// A helper class to avoid long lines accessing mutable configuration.
class ModelConfigMutator {
 public:
  explicit ModelConfigMutator(ModelConfig* const config) : config_(config) {}

  // Sets or updates label and its version.
  ModelConfigMutator SetLabelVersion(const string& label, const int& version) {
    (*config_->mutable_version_labels())[label] = version;
    return *this;
  }

 private:
  ModelConfig* config_;
};

inline ModelConfigMutator MutateModelConfig(ModelServerConfig* const config) {
  return ModelConfigMutator(
      config->mutable_model_config_list()->mutable_config(0));
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_SERVER_CORE_TEST_UTIL_H_
