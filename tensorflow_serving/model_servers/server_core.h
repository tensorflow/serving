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

// ServerCore contains state and helper methods enabling the building of
// ModelServers that support multiple interfaces. All functionality in
// ServerCore is independent of any domain specific APIs and independent of
// platforms.
//
// In terms of state, ServerCore is initialized with and retains a static
// ModelServerConfig, from which it bootstraps an AspiredVersionsManager and
// auxiliary data structures to support efficient serving.
//
// Interfaces built above ServerCore, e.g. RPC service implementations, will
// remain stateless and will perform all lookups of servables (models) via
// ServerCore.

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {

namespace test_util {
class ServerCoreTestAccess;
}  // namespace test_util

class ServerCore : public Manager {
 public:
  using ModelServerSourceAdapter =
      SourceAdapter<StoragePath, std::unique_ptr<Loader>>;

  using SourceAdapterCreator =
      std::function<Status(const string& platform_type,
                           std::unique_ptr<ModelServerSourceAdapter>* adapter)>;

  using ServableStateMonitorCreator =
      std::function<Status(EventBus<ServableState>* event_bus,
                           std::unique_ptr<ServableStateMonitor>* monitor)>;

  // A function that's responsible for instantiating and connecting the
  // necessary custom sources and source adapters to the manager based on a
  // passed in config (any).
  // The expected pattern is that ownership of the created sources/source
  // adapters can be transferred to the manager.
  using CustomModelConfigLoader = std::function<Status(
      const ::google::protobuf::Any& any, EventBus<ServableState>* event_bus,
      UniquePtrWithDeps<AspiredVersionsManager>* manager)>;

  struct Options {
    // ModelServer configuration.
    ModelServerConfig model_server_config;

    // The AspiredVersionPolicy to use for the manager. Must be non-null.
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy;

    // The number of threads used to load and unload models. If set to 0, then
    // no thread pool is used and the loads/unloads are performed serially in
    // the manager thread.
    int32 num_load_unload_threads = 0;

    // The number of load/unload threads used to load the initial set of models
    // at server startup. This is set high to load up the initial set of models
    // fast, after this the server uses num_load_unload_threads.
    int32 num_initial_load_unload_threads = 4.0 * port::NumSchedulableCPUs();

    // Total model size limit, in terms of main memory, in bytes.
    uint64 total_model_memory_limit_bytes = std::numeric_limits<uint64>::max();

    // Maximum number of times we retry loading a model, after the first
    // failure, before we give up.
    int32 max_num_load_retries = 5;

    // Time interval between file-system polls, in seconds.
    int32 file_system_poll_wait_seconds = 30;

    // A function for creating ModelServerSourceAdapter based on the
    // 'platform_type'.
    // If not specified, a default creator that creates
    // SessionBundleSourceAdapter for TensorFlow will be used.
    SourceAdapterCreator source_adapter_creator;

    // A function for creating ServableStateMonitor. If not specified, a default
    // creator that creates ServableStateMonitor will be used.
    ServableStateMonitorCreator servable_state_monitor_creator;

    // A function for instantiating and connecting custom sources and source
    // adapters to the manager.
    CustomModelConfigLoader custom_model_config_loader;
  };

  virtual ~ServerCore() = default;

  // Creates a ServerCore instance with all the models and sources per the
  // ModelServerConfig.
  //
  // For models statically configured with ModelConfigList, waits for them
  // to be made available (or hit an error) for serving before returning.
  // Returns an error status if any such model fails to load.
  static Status Create(Options options, std::unique_ptr<ServerCore>* core);

  std::vector<ServableId> ListAvailableServableIds() const override {
    return manager_->ListAvailableServableIds();
  }

  // Updates the server core with all the models and sources per the
  // ModelServerConfig. Like Create(), waits for all statically configured
  // servables to be made available before returning, and returns an error if
  // any such model fails to load.
  //
  // IMPORTANT: It is only legal to call this method more than once if using
  // ModelConfigList (versus custom model config).
  virtual Status ReloadConfig(const ModelServerConfig& config)
      LOCKS_EXCLUDED(config_mu_);

  // Returns ServableStateMonitor that can be used to query servable states.
  virtual const ServableStateMonitor* servable_state_monitor() const {
    return servable_state_monitor_.get();
  }

  // Returns a ServableHandle given a ServableRequest. Returns error if no such
  // Servable is available -- e.g. not yet loaded, has been quiesced/unloaded,
  // etc. Callers may assume that an OK status indicates a non-null handle.
  //
  // IMPORTANT: The caller should only hold on to a handle for a short time, for
  // example for the duration of a single request. Holding a handle for a long
  // period of time will prevent servable loading and unloading.
  template <typename T>
  Status GetServableHandle(const ModelSpec& model_spec,
                           ServableHandle<T>* const handle) {
    ServableRequest servable_request;
    ServableRequestFromModelSpec(model_spec, &servable_request);
    const tensorflow::Status status =
        manager_->GetServableHandle(servable_request, handle);
    if (!status.ok()) {
      VLOG(1) << "Unable to get servable handle due to: " << status;
      return status;
    }
    return Status::OK();
  }

 protected:
  explicit ServerCore(Options options);

 private:
  friend class test_util::ServerCoreTestAccess;

  // ************************************************************************
  // Server Setup and Initialization.
  // ************************************************************************

  // Initializes server core.
  // Must be run once and only once per ServerCore instance.
  Status Initialize(
      std::unique_ptr<AspiredVersionPolicy> aspired_version_policy);

  // Creates a AspiredVersionsManager with the specified policy.
  Status CreateAspiredVersionsManager(
      std::unique_ptr<AspiredVersionPolicy> policy,
      std::unique_ptr<AspiredVersionsManager>* manager);

  // Creates a ResourceTracker.
  Status CreateResourceTracker(
      std::unique_ptr<ResourceTracker>* resource_tracker);

  // Creates a platform-specific Loader Source.
  Status CreateSourceAdapter(
      const string& model_platform,
      std::unique_ptr<ModelServerSourceAdapter>* adapter);

  // Creates a FileSystemStoragePathSourceConfig from the ModelConfigList of
  // 'config'.
  FileSystemStoragePathSourceConfig CreateStoragePathSourceConfig(
      const ModelServerConfig& config) const;

  // Waits for all models from the ModelConfigList in 'config_' to be loaded.
  // Returns an error if any configured model fails to load.
  Status WaitUntilConfiguredModelsAvailable()
      EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Creates a FileSystemStoragePathSource, connects it to the supplied target,
  // stores the pointer in 'storage_path_source_' and transfers the ownership to
  // 'manager_'.
  Status CreateFileSystemStoragePathSource(
      const FileSystemStoragePathSourceConfig& source_config,
      Target<StoragePath>* target) EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Updates the 'storage_path_source_' config.
  Status ReloadFileSystemStoragePathSourceConfig(
      const FileSystemStoragePathSourceConfig& source_config)
      EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Adds/reloads models through ModelConfigList of 'config_'.
  Status AddModelsViaModelConfigList() EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Adds/reloads models through custom model config of 'config_'.
  Status AddModelsViaCustomModelConfig() EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // ************************************************************************
  // Request Processing.
  // ************************************************************************

  // Extracts a ServableRequest from the given ModelSpec.
  Status ServableRequestFromModelSpec(const ModelSpec& model_spec,
                                      ServableRequest* servable_request) const;

  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) override {
    return manager_->GetUntypedServableHandle(request, untyped_handle);
  }

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override {
    return manager_->GetAvailableUntypedServableHandles();
  }

  // The options passed to the ctor, minus the AspiredVersionPolicy.
  Options options_;

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  std::shared_ptr<ServableStateMonitor> servable_state_monitor_;
  UniquePtrWithDeps<AspiredVersionsManager> manager_;

  // The most recent config supplied to ReloadConfig().
  ModelServerConfig config_ GUARDED_BY(config_mu_);

  // Model platform of the source adapter created for ModelConfigList.
  // Empty if the source adapter is not yet created.
  string model_platform_ GUARDED_BY(config_mu_);

  // If the configuration uses a file-system source, this is populated with a
  // pointer to the source (to enable reconfiguration later). The source is
  // owned by 'manager_'.
  FileSystemStoragePathSource* storage_path_source_ GUARDED_BY(config_mu_) =
      nullptr;

  // A mutex for reconfiguration, used by ReloadConfig().
  mutex config_mu_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
