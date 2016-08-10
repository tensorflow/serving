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
// In terms of state, ServerCore bootstraps with an AspiredVersionsManager to
// support efficient serving. It will soon support (re)loading of
// ModelServerConfig, from which it (re)creates auxiliary data structures to
// load model from custom sources.
//
// Interfaces built above ServerCore, e.g. RPC service implementations, will
// remain stateless and will perform all lookups of servables (models) via
// ServerCore.

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_

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
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {

namespace test_util {
class ServerCoreTestAccess;
}  // namespace test_util

class ServerCore {
 public:
  virtual ~ServerCore() = default;

  using ModelServerSourceAdapter =
      SourceAdapter<StoragePath, std::unique_ptr<Loader>>;

  using SourceAdapterCreator =
      std::function<Status(const string& model_type,
                           std::unique_ptr<ModelServerSourceAdapter>* adapter)>;

  using ServableStateMonitorCreator =
      std::function<Status(EventBus<ServableState>* event_bus,
                           std::unique_ptr<ServableStateMonitor>* monitor)>;

  using DynamicModelConfigLoader = std::function<Status(
      const ::google::protobuf::Any& any,
      Target<std::unique_ptr<Loader>>* target)>;

  // Creates server core and loads the given config.
  //
  // The config is allowed to be empty, in which case user should call
  // ReloadConfig later to actually start the server.
  //
  // source_adapter_creator is used, upon each ReloadConfig, to (re)create the
  // single instance of global source adapter that adapts all Sources to
  // platform-specific Loaders for the global AspiredVersionsManager.
  // servable_state_monitor_creator is used once to create the
  // ServableStateMonitor for the global AspiredVersionsManager.
  // dynamic_model_config_loader is used, upon each ReloadConfig, to (re)create
  // Sources defined in dynamic_model_config Any proto and connect them to the
  // global source adapter.
  static Status Create(
      const ModelServerConfig& config,
      SourceAdapterCreator source_adapter_creator,
      ServableStateMonitorCreator servable_state_monitor_creator,
      DynamicModelConfigLoader dynamic_model_config_loader,
      std::unique_ptr<ServerCore>* core);

  // Updates the server core with all the models/sources per the
  // ModelServerConfig.
  //
  // For static config given as ModelConfigList, it waits for the models to be
  // made available for serving before returning from this method.
  //
  // TODO(b/29012372): Note: this method may be called only when the server
  // currently contains no models.
  virtual Status ReloadConfig(const ModelServerConfig& config);

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
    TF_RETURN_IF_ERROR(manager_->GetServableHandle(servable_request, handle));
    return Status::OK();
  }

 protected:
  explicit ServerCore(
      SourceAdapterCreator source_adapter_creator_,
      ServableStateMonitorCreator servable_state_monitor_creator,
      DynamicModelConfigLoader dynamic_model_config_loader);

 private:
  friend class test_util::ServerCoreTestAccess;

  // ************************************************************************
  // Server Setup and Initialization.
  // ************************************************************************

  // Initializes server core.
  // Must be run once and only once per ServerCore instance.
  Status Initialize();

  // Creates a platform-specific Loader Source by adapting the underlying
  // FileSystemStoragePathSource and connects it to the supplied target.
  Status CreateSourceAdapter(
      const string& model_type, Target<std::unique_ptr<Loader>>* target,
      std::unique_ptr<ModelServerSourceAdapter>* adapter);

  // Creates a Source<StoragePath> that monitors a filesystem's base_path for
  // new directories and connects it to the supplied target.
  // The servable_name param simply allows this source to create all
  // AspiredVersions for the target with the same servable_name.
  Status CreateStoragePathSource(
      const string& base_path, const string& servable_name,
      Target<StoragePath>* target,
      std::unique_ptr<Source<StoragePath>>* path_source);

  // Creates a AspiredVersionsManager with the EagerLoadPolicy.
  Status CreateAspiredVersionsManager(
      std::unique_ptr<AspiredVersionsManager>* manager);

  // Adds models through ModelConfigList, and waits for them to be loaded.
  Status AddModelsViaModelConfigList(const ModelServerConfig& config);

  // Adds models through dynamic model config defined in Any proto.
  Status AddModelsViaDynamicModelConfig(const ModelServerConfig& config);

  // ************************************************************************
  // Request Processing.
  // ************************************************************************

  // Extracts a ServableRequest from the given ModelSpec.
  Status ServableRequestFromModelSpec(const ModelSpec& model_spec,
                                      ServableRequest* servable_request) const;

  // ************************************************************************
  // Test Access.
  // ************************************************************************

  // Lists available servable ids from the wrapped aspired-versions-manager.
  std::vector<ServableId> ListAvailableServableIds() const;

  SourceAdapterCreator source_adapter_creator_;
  ServableStateMonitorCreator servable_state_monitor_creator_;
  DynamicModelConfigLoader dynamic_model_config_loader_;

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  std::shared_ptr<ServableStateMonitor> servable_state_monitor_;
  UniquePtrWithDeps<AspiredVersionsManager> manager_;

  bool seen_models_ GUARDED_BY(seen_models_mu_) = false;
  mutable mutex seen_models_mu_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
