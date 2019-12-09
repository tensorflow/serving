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

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

namespace {

void AddSessionRunLoadThreadPool(SessionBundleConfig* const bundle_config) {
  auto* const session_config = bundle_config->mutable_session_config();
  session_config->add_session_inter_op_thread_pool();
  // The second pool will be used for loading.
  session_config->add_session_inter_op_thread_pool()->set_num_threads(4);
  bundle_config->mutable_session_run_load_threadpool_index()->set_value(1);
}

ServerCore::Options GetDefaultOptions(const bool use_saved_model) {
  ServerCore::Options options;
  options.file_system_poll_wait_seconds = 1;
  // Reduce the number of initial load threads to be num_load_threads to avoid
  // timing out in tests.
  options.num_initial_load_threads = options.num_load_threads;
  options.aspired_version_policy =
      std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
  options.custom_model_config_loader =
      [](const ::google::protobuf::Any& any, EventBus<ServableState>* event_bus,
         UniquePtrWithDeps<AspiredVersionsManager>* manager) -> Status {
    return Status::OK();
  };

  SessionBundleConfig bundle_config;
  AddSessionRunLoadThreadPool(&bundle_config);

  options.platform_config_map =
      CreateTensorFlowPlatformConfigMap(bundle_config, use_saved_model);
  ::google::protobuf::Any fake_source_adapter_config;
  fake_source_adapter_config.PackFrom(
      test_util::FakeLoaderSourceAdapterConfig());
  (*(*options.platform_config_map.mutable_platform_configs())[kFakePlatform]
        .mutable_source_adapter_config()) = fake_source_adapter_config;

  return options;
}

}  // namespace

Status CreateServerCore(const ModelServerConfig& config,
                        ServerCore::Options options,
                        std::unique_ptr<ServerCore>* server_core) {
  options.model_server_config = config;
  return ServerCore::Create(std::move(options), server_core);
}

Status CreateServerCore(const ModelServerConfig& config,
                        std::unique_ptr<ServerCore>* server_core) {
  return CreateServerCore(config, GetDefaultOptions(true /*use_saved_model */),
                          server_core);
}

ModelServerConfig ServerCoreTest::GetTestModelServerConfigForFakePlatform() {
  ModelServerConfig config = GetTestModelServerConfigForTensorflowPlatform();
  ModelConfig* model_config =
      config.mutable_model_config_list()->mutable_config(0);
  model_config->set_model_platform(kFakePlatform);
  return config;
}

ModelServerConfig
ServerCoreTest::GetTestModelServerConfigForTensorflowPlatform() {
  ModelServerConfig config;
  auto model = config.mutable_model_config_list()->add_config();
  model->set_name(kTestModelName);
  if (GetTestType() == SAVED_MODEL) {
    model->set_base_path(test_util::TensorflowTestSrcDirPath(
        "/cc/saved_model/testdata/half_plus_two"));
  } else {
    model->set_base_path(test_util::TestSrcDirPath(
        "/servables/tensorflow/google/testdata/half_plus_two"));
  }
  if (PrefixPathsWithURIScheme()) {
    model->set_base_path(io::CreateURI("file", "", model->base_path()));
  }
  model->set_model_platform(kTensorFlowModelPlatform);
  return config;
}

void ServerCoreTest::SwitchToHalfPlusTwoWith2Versions(
    ModelServerConfig* config) {
  CHECK_EQ(1, config->model_config_list().config().size());
  auto model = config->mutable_model_config_list()->mutable_config(0);
  if (GetTestType() == SAVED_MODEL) {
    model->set_base_path(test_util::TestSrcDirPath(
        "/servables/tensorflow/testdata/saved_model_half_plus_two_2_versions"));
  } else {
    model->set_base_path(test_util::TestSrcDirPath(
        "/servables/tensorflow/google/testdata/half_plus_two_2_versions"));
  }
  // Request loading both versions simultaneously.
  model->clear_model_version_policy();
  model->mutable_model_version_policy()->mutable_all();
  if (PrefixPathsWithURIScheme()) {
    model->set_base_path(io::CreateURI("file", "", model->base_path()));
  }
}

ServerCore::Options ServerCoreTest::GetDefaultOptions() {
  // Model platforms.
  const TestType test_type = GetTestType();
  const bool use_saved_model = test_type == SAVED_MODEL ||
                               test_type == SAVED_MODEL_BACKWARD_COMPATIBILITY;
  return test_util::GetDefaultOptions(use_saved_model);
}

Status ServerCoreTest::CreateServerCore(
    const ModelServerConfig& config, ServerCore::Options options,
    std::unique_ptr<ServerCore>* server_core) {
  return test_util::CreateServerCore(config, std::move(options), server_core);
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
