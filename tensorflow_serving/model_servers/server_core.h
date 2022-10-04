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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/any.pb.h"
#include "absl/base/macros.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/dynamic_source_router.h"
#include "tensorflow_serving/core/prefix_storage_path_source_adapter.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/server_request_logger.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {

namespace test_util {
class ServerCoreTestAccess;
}  // namespace test_util

/// ServerCore contains state and helper methods enabling the building of
/// ModelServers that support multiple interfaces. All functionality in
/// ServerCore is independent of any domain specific APIs and independent of
/// platforms.
///
/// In terms of state, ServerCore is initialized with and retains a static
/// ModelServerConfig, from which it bootstraps an AspiredVersionsManager and
/// auxiliary data structures to support efficient serving.
///
/// Interfaces built above ServerCore, e.g. RPC service implementations, will
/// remain stateless and will perform all lookups of servables (models) via
/// ServerCore.
class ServerCore : public Manager {
 public:
  using PreLoadHook = AspiredVersionsManager::PreLoadHook;

  using ServableStateMonitorCreator =
      std::function<Status(EventBus<ServableState>* event_bus,
                           std::unique_ptr<ServableStateMonitor>* monitor)>;

  /// A function that's responsible for instantiating and connecting the
  /// necessary custom sources and source adapters to the manager based on a
  /// passed in config (any).
  /// The expected pattern is that ownership of the created sources/source
  /// adapters can be transferred to the manager.
  using CustomModelConfigLoader = std::function<Status(
      const ::google::protobuf::Any& any, EventBus<ServableState>* event_bus,
      UniquePtrWithDeps<AspiredVersionsManager>* manager)>;

  /// Function signature used to update the server_request_logger.
  using ServerRequestLoggerUpdater =
      std::function<Status(const ModelServerConfig&, ServerRequestLogger*)>;

  /// Options for configuring a ServerCore object.
  struct Options {
    // ModelServer configuration.
    ModelServerConfig model_server_config;
    // Relative (non-absolute) base-paths in model_server_config will
    // be prepended with model_config_list_root_dir.
    absl::optional<string> model_config_list_root_dir;

    // The AspiredVersionPolicy to use for the manager. Must be non-null.
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy;

    // The number of threads used to load models. If set to 0, then no thread
    // pool is used and loads are performed serially in the manager thread.
    int32 num_load_threads = 0;

    // The number of load threads used to load the initial set of models at
    // server startup. This is set high to load up the initial set of models
    // fast, after this the server uses num_load_threads.
    int32 num_initial_load_threads = 4.0 * port::NumSchedulableCPUs();

    // The number of threads used to unload models. If set to 0, then no thread
    // pool is used and unloads are performed serially in the manager thread.
    int32 num_unload_threads = 0;

    // Total model size limit, in terms of main memory, in bytes.
    uint64_t total_model_memory_limit_bytes =
        std::numeric_limits<uint64_t>::max();

    // Maximum number of times we retry loading a model, after the first
    // failure, before we give up.
    //
    // If set to 0, a load is attempted only once.
    int32 max_num_load_retries = 5;

    // The interval, in microseconds, between each servable load retry. If set
    // negative, we don't wait.
    // Default: 1 minute.
    int64_t load_retry_interval_micros = 1LL * 60 * 1000 * 1000;

    // Time interval between file-system polls, in seconds.
    int32 file_system_poll_wait_seconds = 30;

    // If true, filesystem caches are flushed in the following cases:
    //
    // 1) After the initial models are loaded.
    // 2) After a new config is supplied and a changed set of models are loaded.
    // 3) After each new model version is loaded, if num_load_threads == 1.
    //
    // In the common scenario where the number of load threads is set to 1 after
    // the initial load, this will take care of flushing the cache once after
    // the initial load, and after every subsequent load of every model version.
    bool flush_filesystem_caches = false;

    // Configuration for the supported platforms.
    PlatformConfigMap platform_config_map;

    // A function for creating ServableStateMonitor. If not specified, a default
    // creator that creates ServableStateMonitor will be used.
    ServableStateMonitorCreator servable_state_monitor_creator;

    // A function for instantiating and connecting custom sources and source
    // adapters to the manager.
    CustomModelConfigLoader custom_model_config_loader;

    // Whether to permit incoming ModelSpec requests to use the 'version_label'
    // field.
    bool allow_version_labels = true;

    // If set to true, the server will fail to start up (or fail a config
    // reload) if, for any configured model, no versions of the model are found
    // in the filesystem under the model's base path.
    ABSL_DEPRECATED("Use servable_versions_always_present.")
    bool fail_if_no_model_versions_found = false;

    // For servables which end with LoaderHarness::State::kError, enable
    // future attempts at reload to progress.
    bool enable_reload_servables_with_error = false;

    // If set to true, the server will fail to start up (or fail a config
    // reload) if, for any configured model, no versions of the model are found
    // in the filesystem under the model's base path. In addition, if the
    // filesystem polling finds no servables under the base path for a
    // configured model, it will do nothing, rather than unloading all versions.
    bool servable_versions_always_present = false;

    // Logger used for logging requests hitting the server.
    std::unique_ptr<ServerRequestLogger> server_request_logger;

    // If set, we use this function to update the server_request_logger.
    ServerRequestLoggerUpdater server_request_logger_updater;

    // Callback to be called just before a servable is to be loaded. This will
    // called on the same manager load thread which starts the load.
    PreLoadHook pre_load_hook;

    // Whether to allow assigning unused version labels to models that are not
    // available yet.
    bool allow_version_labels_for_unavailable_models = false;

    // Whether to force-allow assigning any version labels to models that are
    // not available yet.
    bool force_allow_any_version_labels_for_unavailable_models = false;

    // In a predict handler, this option specifies how to serialize tensors
    // (e.g: as proto fields or as proto content).
    // Serialize as proto fields by default, for backward compatibility.
    internal::PredictResponseTensorSerializationOption
        predict_response_tensor_serialization_option =
            internal::PredictResponseTensorSerializationOption::kAsProtoField;

    // The prefix to append to the file system storage paths.
    std::string storage_path_prefix;

    bool enable_cors_support = false;
  };

  virtual ~ServerCore() = default;

  /// Creates a ServerCore instance with all the models and sources per the
  /// ModelServerConfig.
  ///
  /// For models statically configured with ModelConfigList, waits for them
  /// to be made available (or hit an error) for serving before returning.
  /// Returns an error status if any such model fails to load.
  static Status Create(Options options, std::unique_ptr<ServerCore>* core);

  std::vector<ServableId> ListAvailableServableIds() const override {
    return manager_->ListAvailableServableIds();
  }

  /// Updates the server core with all the models and sources per the
  /// ModelServerConfig. Like Create(), waits for all statically configured
  /// servables to be made available before returning, and returns an error if
  /// any such model fails to load. (Does not necessarily wait for models
  /// removed from the config to finish unloading; that may occur
  /// asynchronously.)
  ///
  /// IMPORTANT: It is only legal to call this method more than once if using
  /// ModelConfigList (versus custom model config).
  virtual Status ReloadConfig(const ModelServerConfig& config)
      TF_LOCKS_EXCLUDED(config_mu_);

  /// Returns ServableStateMonitor that can be used to query servable states.
  virtual ServableStateMonitor* servable_state_monitor() const {
    return servable_state_monitor_.get();
  }

  /// Returns a ServableHandle given a ModelSpec. Returns error if no such
  /// Servable is available -- e.g. not yet loaded, has been quiesced/unloaded,
  /// etc. Callers may assume that an OK status indicates a non-null handle.
  ///
  /// IMPORTANT: The caller should only hold on to a handle for a short time,
  /// for example for the duration of a single request. Holding a handle for a
  /// long period of time will prevent servable loading and unloading.
  ///
  /// If 'options_.allow_version_labels==true', recognizes two specific model
  /// version labels -- "stable" and "canary" -- and resolves them to the
  /// smallest and largest available version, respectively.
  template <typename T>
  Status GetServableHandle(const ModelSpec& model_spec,
                           ServableHandle<T>* const handle) {
    ServableRequest servable_request;
    tensorflow::Status status =
        ServableRequestFromModelSpec(model_spec, &servable_request);
    if (!status.ok()) {
      VLOG(1) << "Unable to get servable handle due to: " << status;
      return status;
    }
    status = manager_->GetServableHandle(servable_request, handle);
    if (!status.ok()) {
      VLOG(1) << "Unable to get servable handle due to: " << status;
      return status;
    }
    return Status();
  }

  /// Writes the log for the particular request, response and metadata, if we
  /// decide to sample it and if request-logging was configured for the
  /// particular model.
  virtual Status Log(const google::protobuf::Message& request,
                     const google::protobuf::Message& response,
                     const LogMetadata& log_metadata) {
    return options_.server_request_logger->Log(request, response, log_metadata);
  }

  internal::PredictResponseTensorSerializationOption
  predict_response_tensor_serialization_option() const {
    return options_.predict_response_tensor_serialization_option;
  }

  bool enable_cors_support() const { return options_.enable_cors_support; }

 protected:
  ServerCore(Options options);

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

  // Creates a platform-specific source adapter.
  Status CreateAdapter(
      const string& model_platform,
      std::unique_ptr<StoragePathSourceAdapter>* adapter) const;

  // Creates a FileSystemStoragePathSourceConfig from the ModelConfigList of
  // 'config'.
  FileSystemStoragePathSourceConfig CreateStoragePathSourceConfig(
      const ModelServerConfig& config) const;

  // Creates routes for a DynamicSourceRouter from the ModelConfigList of
  // 'config'.
  Status CreateStoragePathRoutes(
      const ModelServerConfig& config,
      DynamicSourceRouter<StoragePath>::Routes* routes) const;

  // Waits until all entries in 'models' have been loaded, according to
  // 'monitor'. Returns an error if any model fails to load.
  Status WaitUntilModelsAvailable(const std::set<string>& models,
                                  ServableStateMonitor* monitor);

  // Creates a FileSystemStoragePathSource and an optional
  // PrefixStoragePathSourceAdapter, and connects them to the supplied target.
  Status CreateStoragePathSource(
      const FileSystemStoragePathSourceConfig& config,
      Target<StoragePath>* target,
      std::unique_ptr<FileSystemStoragePathSource>* source,
      std::unique_ptr<PrefixStoragePathSourceAdapter>* prefix_source_adapter)
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // The source adapters to deploy, to handle the configured platforms as well
  // as models whose platform is unknown (errors).
  //
  // Importantly, we deploy one source adapter per platform, not one per model,
  // to handle cross-model optimizations that some platforms/adapters may employ
  // e.g. cross-model batch scheduling.
  struct SourceAdapters {
    // One adapter for each platform.
    std::map<string, std::unique_ptr<StoragePathSourceAdapter>>
        platform_adapters;

    // An extra adapter to report errors for models with no configured platform.
    std::unique_ptr<StoragePathSourceAdapter> error_adapter;
  };

  // Creates a source router and connects it to the supplied adapter targets.
  Status CreateRouter(
      const DynamicSourceRouter<StoragePath>::Routes& routes,
      SourceAdapters* targets,
      std::unique_ptr<DynamicSourceRouter<StoragePath>>* router) const;

  // Creates a set of source adapters based on options_.platform_config_map.
  Status CreateAdapters(SourceAdapters* adapters) const;

  // Connects the source adapters to the manager and waits it to load all
  // configured models.
  Status ConnectAdaptersToManagerAndAwaitModelLoads(SourceAdapters* adapters)
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Updates the config of 'storage_path_source_and_router_->source'.
  Status ReloadStoragePathSourceConfig(
      const FileSystemStoragePathSourceConfig& source_config)
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Updates the configured routes of 'storage_path_source_and_router_->router'.
  Status ReloadRoutes(const DynamicSourceRouter<StoragePath>::Routes& routes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Adds/reloads models through ModelConfigList of 'config_'.
  Status AddModelsViaModelConfigList() TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Adds/reloads models through custom model config of 'config_'.
  Status AddModelsViaCustomModelConfig()
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Updates the ServerRequestLogger based on the ModelConfigList.
  Status MaybeUpdateServerRequestLogger(
      ModelServerConfig::ConfigCase config_case)
      TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_);

  // Updates 'model_labels_to_versions_' based on 'config_'. Throws an error if
  // requesting to assign an existing label to a version not in state
  // kAvailable. For a new version label, it can be assigned to a version that
  // is not in state kAvailable yet if
  // allow_version_labels_for_unavailable_models is true.
  Status UpdateModelVersionLabelMap() TF_EXCLUSIVE_LOCKS_REQUIRED(config_mu_)
      TF_LOCKS_EXCLUDED(model_labels_to_versions_mu_);

  // ************************************************************************
  // Request Processing.
  // ************************************************************************

  // Extracts a ServableRequest from the given ModelSpec.
  Status ServableRequestFromModelSpec(const ModelSpec& model_spec,
                                      ServableRequest* servable_request) const;

  // Gets the version associated with 'label', for the given model name.
  Status GetModelVersionForLabel(const string& model_name, const string& label,
                                 int64_t* version) const
      TF_LOCKS_EXCLUDED(model_labels_to_versions_mu_);

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

  // All of the supported platforms (i.e. the ones given in
  // 'options_.platform_config_map'), and a router output port number for each.
  // Used to deterministically associate a platform with a source adapter.
  std::map<string, int> platform_to_router_port_;

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  std::shared_ptr<ServableStateMonitor> servable_state_monitor_;
  UniquePtrWithDeps<AspiredVersionsManager> manager_;

  // The most recent config supplied to ReloadConfig().
  ModelServerConfig config_ TF_GUARDED_BY(config_mu_);

  // A model_name->label->version# map.
  std::unique_ptr<std::map<string, std::map<string, int64_t>>>
      model_labels_to_versions_ TF_GUARDED_BY(model_labels_to_versions_mu_);

  struct StoragePathSourceAndRouter {
    FileSystemStoragePathSource* source;
    DynamicSourceRouter<StoragePath>* router;
  };

  // If the configuration uses a file-system source, this is populated with
  // pointers to the source and router (to enable reconfiguration later). Both
  // are owned by 'manager_'.
  absl::optional<StoragePathSourceAndRouter> storage_path_source_and_router_
      TF_GUARDED_BY(config_mu_);

  // A mutex for reconfiguration, used by ReloadConfig().
  mutable mutex config_mu_;

  // A mutex for swapping the model version label map. Should only be held for
  // a short time (i.e. pointer swap) to avoid holding up inference requests.
  mutable mutex model_labels_to_versions_mu_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_CORE_H_
