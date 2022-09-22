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

#include "tensorflow_serving/model_servers/server_core.h"

#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/config/file_system_storage_path_source.pb.h"
#include "tensorflow_serving/core/load_servables_fast.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"

namespace tensorflow {
namespace serving {

// ************************************************************************
// Local Helper Methods.
// ************************************************************************

namespace {

// Gets the platform associated with a model.
Status GetPlatform(const ModelConfig& model_config, string* platform) {
  if (model_config.model_type() != ModelType::MODEL_TYPE_UNSPECIFIED) {
    LOG(WARNING) << "Deprecated ModelServerConfig::model_type field used. "
                    "Prefer ModelServerConfig::model_platform.";
    if (!model_config.model_platform().empty()) {
      return errors::InvalidArgument(
          "Illegal setting both ModelServerConfig::model_type (deprecated) "
          "and ModelServerConfig::model_platform.");
    }
    if (model_config.model_type() == ModelType::TENSORFLOW) {
      *platform = kTensorFlowModelPlatform;
    } else {
      return errors::InvalidArgument(
          strings::StrCat("ModelServerConfig::model_type choice ",
                          model_config.model_type(), " not supported."));
    }
  } else {
    *platform = model_config.model_platform();
  }

  if (platform->empty()) {
    return errors::InvalidArgument(
        "Illegal setting neither ModelServerConfig::model_type (deprecated) "
        "nor ModelServerConfig::model_platform.");
  }
  return OkStatus();
}

// Determines whether a URI is just a relative path.
bool UriIsRelativePath(StringPiece uri) {
  StringPiece scheme, host, path;
  io::ParseURI(uri, &scheme, &host, &path);
  return scheme.empty() && host.empty() && !io::IsAbsolutePath(path);
}

// Returns an error if 'config_list' is invalid in some way, e.g. a model name
// appearing multiple times.
Status ValidateModelConfigList(const ModelConfigList& config_list,
                               const ServerCore::Options& options) {
  // Unique model-names.
  std::set<string> model_names;
  for (const ModelConfig& config : config_list.config()) {
    if (model_names.find(config.name()) != model_names.end()) {
      return errors::InvalidArgument(
          strings::StrCat("Illegal to list model ", config.name(),
                          " multiple times in config list"));
    }
    model_names.insert(config.name());
  }

  // Base-paths are either all relative, or all absolute.
  // WARNING: abuse of terminology!  These "paths" may be URIs :-(
  if (options.model_config_list_root_dir) {
    // All base-paths must be relative.
    if (UriIsRelativePath(*options.model_config_list_root_dir)) {
      return errors::InvalidArgument(
          strings::StrCat("Expected non-empty absolute path or URI; got "
                          "model_config_list_root_dir=",
                          *options.model_config_list_root_dir));
    }
  } else {
    // All base-paths must be absolute.
    for (const ModelConfig& config : config_list.config()) {
      if (UriIsRelativePath(config.base_path())) {
        return errors::InvalidArgument(strings::StrCat(
            "Expected model ", config.name(),
            " to have an absolute path or URI; got base_path()=",
            config.base_path()));
      }
    }
  }

  return OkStatus();
}

// Returns an error if a model exists in both configs, but with different
// platforms.
Status ValidateNoModelsChangePlatforms(const ModelConfigList& old_config_list,
                                       const ModelConfigList& new_config_list) {
  std::map<string, string> old_model_platforms;
  for (const ModelConfig& old_config : old_config_list.config()) {
    string platform;
    TF_RETURN_IF_ERROR(GetPlatform(old_config, &platform));
    old_model_platforms[old_config.name()] = platform;
  }
  for (const ModelConfig& new_config : new_config_list.config()) {
    auto it = old_model_platforms.find(new_config.name());
    if (it == old_model_platforms.end()) {
      continue;
    }
    const string& old_platform = it->second;
    string new_platform;
    TF_RETURN_IF_ERROR(GetPlatform(new_config, &new_platform));
    if (new_platform != old_platform) {
      return errors::InvalidArgument(
          strings::StrCat("Illegal to change a model's platform. For model ",
                          new_config.name(), " platform was ", old_platform,
                          " and new platform requested is ", new_platform));
    }
  }
  return OkStatus();
}

// Unions two route maps. Gives an error if there is a key that is present in
// both 'a' and 'b' but with different values.
Status UnionRoutes(const DynamicSourceRouter<StoragePath>::Routes& a,
                   const DynamicSourceRouter<StoragePath>::Routes& b,
                   DynamicSourceRouter<StoragePath>::Routes* result) {
  *result = a;
  for (const auto& b_entry : b) {
    auto a_it = a.find(b_entry.first);
    if (a_it == a.end()) {
      (*result)[b_entry.first] = b_entry.second;
    } else {
      if (a_it->second != b_entry.second) {
        return errors::InvalidArgument(
            "Conflict while unioning two route maps.");
      }
    }
  }
  return OkStatus();
}

// Finds all models that occur in 'new_config' but not in 'old_config'.
std::set<string> NewModelNamesInSourceConfig(
    const FileSystemStoragePathSourceConfig& old_config,
    const FileSystemStoragePathSourceConfig& new_config) {
  std::set<string> old_models;
  for (const FileSystemStoragePathSourceConfig::ServableToMonitor& servable :
       old_config.servables()) {
    const string& model_name = servable.servable_name();
    old_models.insert(model_name);
  }
  std::set<string> new_models;
  for (const FileSystemStoragePathSourceConfig::ServableToMonitor& servable :
       new_config.servables()) {
    const string& model_name = servable.servable_name();
    if (old_models.find(model_name) == old_models.end()) {
      new_models.insert(model_name);
    }
  }
  return new_models;
}

// Updates the base_path fields in each ModelConfig, prepending an
// absolute model_config_list_root_dir.
// It is assumed that initially, all the base_path fields are relative.
Status UpdateModelConfigListRelativePaths(
    const string& model_config_list_root_dir, ModelConfigList* config_list) {
  std::vector<string> updated_paths;
  updated_paths.reserve(config_list->config_size());
  for (const ModelConfig& config : config_list->config()) {
    // Don't modify absolute paths.
    if (!UriIsRelativePath(config.base_path())) {
      updated_paths.push_back(config.base_path());
      continue;
    }
    updated_paths.emplace_back(
        io::JoinPath(model_config_list_root_dir, config.base_path()));
    if (UriIsRelativePath(updated_paths.back())) {
      return errors::InvalidArgument(strings::StrCat(
          "Expected model ", config.name(),
          " with updated base_path = JoinPath(", model_config_list_root_dir,
          ", ", config.base_path(), ") to have an absolute path; got ",
          updated_paths.back()));
    }
  }
  for (int ii = 0; ii < updated_paths.size(); ++ii) {
    config_list->mutable_config(ii)->set_base_path(updated_paths[ii]);
  }
  return OkStatus();
}

}  // namespace

// ************************************************************************
// Public Methods.
// ************************************************************************

Status ServerCore::Create(Options options,
                          std::unique_ptr<ServerCore>* server_core) {
  if (options.servable_state_monitor_creator == nullptr) {
    options.servable_state_monitor_creator =
        [](EventBus<ServableState>* event_bus,
           std::unique_ptr<ServableStateMonitor>* monitor) {
          monitor->reset(new ServableStateMonitor(event_bus));
          return OkStatus();
        };
  }

  if (options.server_request_logger == nullptr) {
    TF_RETURN_IF_ERROR(
        ServerRequestLogger::Create(nullptr, &options.server_request_logger));
  }

  // We need to move the aspired_version_policy first because we will move the
  // server_core_config (which contains aspired_version_policy) below.
  std::unique_ptr<AspiredVersionPolicy> aspired_version_policy =
      std::move(options.aspired_version_policy);
  auto model_server_config = options.model_server_config;
  server_core->reset(new ServerCore(std::move(options)));
  TF_RETURN_IF_ERROR(
      (*server_core)->Initialize(std::move(aspired_version_policy)));
  return (*server_core)->ReloadConfig(model_server_config);
}

// ************************************************************************
// Server Setup and Initialization.
// ************************************************************************

ServerCore::ServerCore(Options options)
    : options_(std::move(options)),
      servable_event_bus_(EventBus<ServableState>::CreateEventBus()) {
  // Number the platforms. (The proto map iteration order is nondeterministic,
  // but we don't care since the numbering is arbitrary.)
  int port_num = 0;
  for (const auto& entry : options_.platform_config_map.platform_configs()) {
    const string& platform = entry.first;
    platform_to_router_port_[platform] = port_num++;
  }
}

Status ServerCore::Initialize(std::unique_ptr<AspiredVersionPolicy> policy) {
  std::unique_ptr<ServableStateMonitor> servable_state_monitor;
  const tensorflow::Status status = options_.servable_state_monitor_creator(
      servable_event_bus_.get(), &servable_state_monitor);
  if (!status.ok()) {
    VLOG(1) << "Servable state monitor creation failed: " << status;
    return status;
  }

  servable_state_monitor_ = std::move(servable_state_monitor);

  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  TF_RETURN_IF_ERROR(CreateAspiredVersionsManager(std::move(policy),
                                                  &aspired_versions_manager));
  manager_.SetOwned(std::move(aspired_versions_manager));

  return OkStatus();
}

Status ServerCore::WaitUntilModelsAvailable(const std::set<string>& models,
                                            ServableStateMonitor* monitor) {
  std::vector<ServableRequest> awaited_servables;
  awaited_servables.reserve(models.size());
  for (const string& model : models) {
    awaited_servables.push_back(ServableRequest::Latest(model));
  }
  std::map<ServableId, ServableState::ManagerState> states_reached;
  const bool all_models_available = monitor->WaitUntilServablesReachState(
      awaited_servables, ServableState::ManagerState::kAvailable,
      &states_reached);
  if (!all_models_available) {
    const int num_unavailable_models = std::count_if(
        states_reached.begin(), states_reached.end(),
        [](const std::pair<ServableId, ServableState::ManagerState>&
               id_and_state) {
          return id_and_state.second != ServableState::ManagerState::kAvailable;
        });
    string message = strings::StrCat(num_unavailable_models,
                                     " model(s) did not become available: {");
    for (const auto& id_and_state : states_reached) {
      if (id_and_state.second != ServableState::ManagerState::kAvailable) {
        absl::optional<ServableState> maybe_state =
            monitor->GetState(id_and_state.first);
        const string error_msg =
            maybe_state && !maybe_state.value().health.ok()
                ? " due to error: " + maybe_state.value().health.ToString()
                : "";
        strings::StrAppend(&message, "{", id_and_state.first.DebugString(),
                           error_msg, "}, ");
      }
    }
    strings::StrAppend(&message, "}");
    return errors::Unknown(message);
  }
  return OkStatus();
}

Status ServerCore::AddModelsViaModelConfigList() {
  const bool is_first_config = storage_path_source_and_router_ == absl::nullopt;

  // Create/reload the source, source router and source adapters.
  const FileSystemStoragePathSourceConfig source_config =
      CreateStoragePathSourceConfig(config_);
  DynamicSourceRouter<StoragePath>::Routes routes;
  TF_RETURN_IF_ERROR(CreateStoragePathRoutes(config_, &routes));
  if (is_first_config) {
    // Construct the following source topology:
    //   Source -> Router -> Adapter_0 (for models using platform 0)
    //                    -> Adapter_1 (for models using platform 1)
    //                    -> ...
    //                    -> ErrorAdapter (for unrecognized models)
    SourceAdapters adapters;
    TF_RETURN_IF_ERROR(CreateAdapters(&adapters));
    std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
    TF_RETURN_IF_ERROR(CreateRouter(routes, &adapters, &router));
    std::unique_ptr<FileSystemStoragePathSource> source;
    std::unique_ptr<PrefixStoragePathSourceAdapter> prefix_source_adapter;
    TF_RETURN_IF_ERROR(CreateStoragePathSource(
        source_config, router.get(), &source, &prefix_source_adapter));

    // Connect the adapters to the manager, and wait for the models to load.
    TF_RETURN_IF_ERROR(ConnectAdaptersToManagerAndAwaitModelLoads(&adapters));

    // Stow the source components.
    storage_path_source_and_router_ = {source.get(), router.get()};
    manager_.AddDependency(std::move(source));
    if (prefix_source_adapter != nullptr) {
      manager_.AddDependency(std::move(prefix_source_adapter));
    }
    manager_.AddDependency(std::move(router));
    for (auto& entry : adapters.platform_adapters) {
      auto& adapter = entry.second;
      manager_.AddDependency(std::move(adapter));
    }
    manager_.AddDependency(std::move(adapters.error_adapter));
  } else {
    // Create a fresh servable state monitor, to avoid getting confused if we're
    // re-loading a model-version that has previously been unloaded.
    ServableStateMonitor fresh_servable_state_monitor(
        servable_event_bus_.get());

    // Figure out which models are new.
    const std::set<string> new_models = NewModelNamesInSourceConfig(
        storage_path_source_and_router_->source->config(), source_config);

    // Now we're ready to start reconfiguring the elements of the Source->
    // Manager pipeline ...

    // First, add the new routes without removing the old ones.
    DynamicSourceRouter<StoragePath>::Routes old_and_new_routes;
    const Status union_status =
        UnionRoutes(storage_path_source_and_router_->router->GetRoutes(),
                    routes, &old_and_new_routes);
    if (!union_status.ok()) {
      // ValidateNoModelsChangePlatforms() should have detected any conflict.
      DCHECK(false);
      return errors::Internal("Old and new routes conflict.");
    }
    TF_RETURN_IF_ERROR(ReloadRoutes(old_and_new_routes));

    // Change the source config. Among other things this will cause it to emit
    // tear-downs of any models that aren't present in the new config.
    TF_RETURN_IF_ERROR(ReloadStoragePathSourceConfig(source_config));

    // Now that any old models are out of the picture, remove the old routes.
    TF_RETURN_IF_ERROR(ReloadRoutes(routes));

    // Wait for any new models to get loaded and become available.
    TF_RETURN_IF_ERROR(
        WaitUntilModelsAvailable(new_models, &fresh_servable_state_monitor));
  }
  return OkStatus();
}

Status ServerCore::AddModelsViaCustomModelConfig() {
  if (options_.custom_model_config_loader == nullptr) {
    return errors::InvalidArgument(
        "Missing custom_model_config_loader in ServerCore Options");
  }

  return options_.custom_model_config_loader(
      config_.custom_model_config(), servable_event_bus_.get(), &manager_);
}

Status ServerCore::MaybeUpdateServerRequestLogger(
    const ModelServerConfig::ConfigCase config_case) {
  if (options_.server_request_logger_updater) {
    return options_.server_request_logger_updater(
        config_, options_.server_request_logger.get());
  }

  if (config_case == ModelServerConfig::kModelConfigList) {
    std::map<string, std::vector<LoggingConfig>> logging_config_map;
    for (const auto& model_config : config_.model_config_list().config()) {
      if (model_config.has_logging_config()) {
        logging_config_map.insert(
            {model_config.name(), {model_config.logging_config()}});
      }
    }
    return options_.server_request_logger->Update(logging_config_map);
  }

  return OkStatus();
}

Status ServerCore::ReloadConfig(const ModelServerConfig& new_config) {
  mutex_lock l(config_mu_);

  // Determine whether to accept this config transition.
  const bool is_first_config =
      config_.config_case() == ModelServerConfig::CONFIG_NOT_SET;
  const bool accept_transition =
      is_first_config ||
      (config_.config_case() == ModelServerConfig::kModelConfigList &&
       new_config.config_case() == ModelServerConfig::kModelConfigList);
  if (!accept_transition) {
    return errors::FailedPrecondition(
        "Cannot transition to requested config. It is only legal to transition "
        "from one ModelConfigList to another.");
  }
  if (new_config.config_case() == ModelServerConfig::CONFIG_NOT_SET) {
    // Nothing to load. In this case we allow a future call with a non-empty
    // config.
    LOG(INFO) << "Taking no action for empty config.";
    return OkStatus();
  }
  if (new_config.config_case() == ModelServerConfig::kModelConfigList) {
    TF_RETURN_IF_ERROR(
        ValidateModelConfigList(new_config.model_config_list(), options_));
  }
  if (new_config.config_case() == ModelServerConfig::kModelConfigList &&
      config_.config_case() == ModelServerConfig::kModelConfigList) {
    TF_RETURN_IF_ERROR(ValidateNoModelsChangePlatforms(
        config_.model_config_list(), new_config.model_config_list()));
  }
  config_ = new_config;

  TF_RETURN_IF_ERROR(UpdateModelVersionLabelMap());

  LOG(INFO) << "Adding/updating models.";
  switch (config_.config_case()) {
    case ModelServerConfig::kModelConfigList: {
      if (options_.model_config_list_root_dir) {
        TF_RETURN_IF_ERROR(UpdateModelConfigListRelativePaths(
            *options_.model_config_list_root_dir,
            config_.mutable_model_config_list()));
      }
      TF_RETURN_IF_ERROR(AddModelsViaModelConfigList());
      break;
    }
    case ModelServerConfig::kCustomModelConfig: {
      // We've already verified this invariant above, so this check should
      // always pass.
      CHECK(is_first_config);  // Crash ok.
      TF_RETURN_IF_ERROR(AddModelsViaCustomModelConfig());
      break;
    }
    default:
      return errors::InvalidArgument("Invalid ServerModelConfig");
  }
  LOG(INFO) << "Finished adding/updating models";

  TF_RETURN_IF_ERROR(MaybeUpdateServerRequestLogger(config_.config_case()));

  if (options_.flush_filesystem_caches) {
    return Env::Default()->FlushFileSystemCaches();
  }

  LOG(INFO) << "Finished reloading config";
  return OkStatus();
}

Status ServerCore::UpdateModelVersionLabelMap() {
  std::unique_ptr<std::map<string, std::map<string, int64_t>>> new_label_map(
      new std::map<string, std::map<string, int64_t>>);
  for (const ModelConfig& model_config : config_.model_config_list().config()) {
    ServableStateMonitor::VersionMap serving_states =
        servable_state_monitor_->GetVersionStates(model_config.name());

    for (const auto& entry : model_config.version_labels()) {
      const string& label = entry.first;
      const int64_t version = entry.second;

      bool allow_any_version_labels_for_unavailable_models = false;
      if (options_.force_allow_any_version_labels_for_unavailable_models) {
        allow_any_version_labels_for_unavailable_models = true;
      } else if (options_.allow_version_labels_for_unavailable_models) {
        int64_t existing_version;
        bool contains_existing_label_with_different_version =
            GetModelVersionForLabel(model_config.name(), label,
                                    &existing_version)
                .ok() &&
            existing_version != version;
        allow_any_version_labels_for_unavailable_models =
            !contains_existing_label_with_different_version;
      }

      // Verify that the label points to a version that is currently available.
      auto serving_states_it = serving_states.find(version);
      if (!allow_any_version_labels_for_unavailable_models &&
          (serving_states_it == serving_states.end() ||
           serving_states_it->second.state.manager_state !=
               ServableState::ManagerState::kAvailable)) {
        return errors::FailedPrecondition(strings::StrCat(
            "Request to assign label to version ", version, " of model ",
            model_config.name(),
            ", which is not currently available for inference"));
      }

      (*new_label_map)[model_config.name()][label] = version;
    }
  }

  if (!options_.allow_version_labels) {
    if (!new_label_map->empty()) {
      return errors::FailedPrecondition(
          "Model version labels are not currently allowed by the server.");
    }
    return OkStatus();
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Updated model label map is: ";
    for (const auto& model_name_and_version_labels : *new_label_map) {
      for (const auto& label_and_version :
           model_name_and_version_labels.second) {
        VLOG(4) << "\t Model name: " << model_name_and_version_labels.first
                << "\t label: " << label_and_version.first
                << " at version: " << label_and_version.second;
      }
    }
  }

  mutex_lock l(model_labels_to_versions_mu_);
  model_labels_to_versions_.swap(new_label_map);

  return OkStatus();
}

Status ServerCore::CreateAdapter(
    const string& model_platform,
    std::unique_ptr<StoragePathSourceAdapter>* adapter) const {
  auto config_it =
      options_.platform_config_map.platform_configs().find(model_platform);
  if (config_it == options_.platform_config_map.platform_configs().end()) {
    return errors::FailedPrecondition(strings::StrCat(
        "PlatformConfigMap has no entry for platform ", model_platform));
  }
  const ::google::protobuf::Any& adapter_config =
      config_it->second.source_adapter_config();
  const tensorflow::Status status =
      StoragePathSourceAdapterRegistry::CreateFromAny(adapter_config, adapter);
  if (!status.ok()) {
    VLOG(1) << "Source adapter creation failed: " << status;
  }
  return status;
}

FileSystemStoragePathSourceConfig ServerCore::CreateStoragePathSourceConfig(
    const ModelServerConfig& config) const {
  FileSystemStoragePathSourceConfig source_config;
  source_config.set_file_system_poll_wait_seconds(
      options_.file_system_poll_wait_seconds);
  source_config.set_fail_if_zero_versions_at_startup(
      options_.fail_if_no_model_versions_found);
  source_config.set_servable_versions_always_present(
      options_.servable_versions_always_present);
  for (const auto& model : config.model_config_list().config()) {
    LOG(INFO) << " (Re-)adding model: " << model.name();
    FileSystemStoragePathSourceConfig::ServableToMonitor* servable =
        source_config.add_servables();
    servable->set_servable_name(model.name());
    servable->set_base_path(model.base_path());
    *servable->mutable_servable_version_policy() = model.model_version_policy();
  }
  return source_config;
}

Status ServerCore::CreateStoragePathRoutes(
    const ModelServerConfig& config,
    DynamicSourceRouter<StoragePath>::Routes* routes) const {
  for (const ModelConfig& model_config : config.model_config_list().config()) {
    const string& model_name = model_config.name();
    string platform;
    TF_RETURN_IF_ERROR(GetPlatform(model_config, &platform));
    auto it = platform_to_router_port_.find(platform);
    if (it == platform_to_router_port_.end()) {
      return errors::InvalidArgument(strings::StrCat(
          "Model ", model_name, " requests unsupported platform ", platform));
    }
    const int port = it->second;
    (*routes)[model_name] = port;
  }
  return OkStatus();
}

Status ServerCore::CreateStoragePathSource(
    const FileSystemStoragePathSourceConfig& config,
    Target<StoragePath>* target,
    std::unique_ptr<FileSystemStoragePathSource>* source,
    std::unique_ptr<PrefixStoragePathSourceAdapter>* prefix_source_adapter) {
  const Status status = FileSystemStoragePathSource::Create(config, source);
  if (!status.ok()) {
    VLOG(1) << "Unable to create FileSystemStoragePathSource due to: "
            << status;
    return status;
  }
  if (options_.storage_path_prefix.empty()) {
    ConnectSourceToTarget(source->get(), target);
  } else {
    *prefix_source_adapter = absl::make_unique<PrefixStoragePathSourceAdapter>(
        options_.storage_path_prefix);
    ConnectSourceToTarget(source->get(), prefix_source_adapter->get());
    ConnectSourceToTarget(prefix_source_adapter->get(), target);
  }
  return OkStatus();
}

Status ServerCore::CreateRouter(
    const DynamicSourceRouter<StoragePath>::Routes& routes,
    SourceAdapters* targets,
    std::unique_ptr<DynamicSourceRouter<StoragePath>>* router) const {
  const int num_output_ports = targets->platform_adapters.size() + 1;
  const Status status = DynamicSourceRouter<StoragePath>::Create(
      num_output_ports, routes, router);
  if (!status.ok()) {
    VLOG(1) << "Unable to create DynamicSourceRouter due to: " << status;
    return status;
  }

  std::vector<Source<StoragePath>*> output_ports = (*router)->GetOutputPorts();
  for (auto& entry : targets->platform_adapters) {
    const string& platform = entry.first;
    StoragePathSourceAdapter* adapter = entry.second.get();

    auto it = platform_to_router_port_.find(platform);
    if (it == platform_to_router_port_.end()) {
      DCHECK(false);
      return errors::Internal("Router port for platform not found.");
    }
    const int port_num = it->second;

    ConnectSourceToTarget(output_ports[port_num], adapter);
  }
  ConnectSourceToTarget(output_ports[output_ports.size() - 1],
                        targets->error_adapter.get());

  return OkStatus();
}

Status ServerCore::CreateAdapters(SourceAdapters* adapters) const {
  for (const auto& entry : platform_to_router_port_) {
    const string& platform = entry.first;
    std::unique_ptr<StoragePathSourceAdapter> adapter;
    TF_RETURN_IF_ERROR(CreateAdapter(platform, &adapter));
    adapters->platform_adapters[platform] = std::move(adapter);
  }
  adapters->error_adapter.reset(
      new ErrorInjectingSourceAdapter<StoragePath, std::unique_ptr<Loader>>(
          errors::Internal("No platform found for model")));
  return OkStatus();
}

Status ServerCore::ConnectAdaptersToManagerAndAwaitModelLoads(
    SourceAdapters* adapters) {
  std::vector<ServableRequest> models_to_await;
  for (const ModelConfig& model_config : config_.model_config_list().config()) {
    models_to_await.push_back(ServableRequest::Latest(model_config.name()));
  }

  std::vector<Source<std::unique_ptr<Loader>>*> adapter_list;
  for (auto& entry : adapters->platform_adapters) {
    adapter_list.push_back(entry.second.get());
  }
  adapter_list.push_back(adapters->error_adapter.get());

  const Status status = ConnectSourcesWithFastInitialLoad(
      manager_.get(), adapter_list, servable_state_monitor_.get(),
      models_to_await, options_.num_initial_load_threads);
  if (!status.ok()) {
    VLOG(1) << "Unable to ConnectSourcesWithFastInitialLoad due to: " << status;
    return status;
  }

  return OkStatus();
}

Status ServerCore::ReloadStoragePathSourceConfig(
    const FileSystemStoragePathSourceConfig& source_config) {
  const Status status =
      storage_path_source_and_router_->source->UpdateConfig(source_config);
  if (!status.ok()) {
    VLOG(1) << "Unable to ReloadStoragePathSourceConfig due to: " << status;
  }
  return status;
}

Status ServerCore::ReloadRoutes(
    const DynamicSourceRouter<StoragePath>::Routes& routes) {
  const Status status =
      storage_path_source_and_router_->router->UpdateRoutes(routes);
  if (!status.ok()) {
    VLOG(1) << "Unable to ReloadRoutes due to: " << status;
  }
  return status;
}

Status ServerCore::CreateAspiredVersionsManager(
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy,
    std::unique_ptr<AspiredVersionsManager>* const manager) {
  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  AspiredVersionsManager::Options manager_options;
  std::unique_ptr<ResourceTracker> resource_tracker;
  TF_RETURN_IF_ERROR(CreateResourceTracker(&resource_tracker));
  manager_options.resource_tracker = std::move(resource_tracker);
  manager_options.servable_event_bus = servable_event_bus_.get();
  manager_options.aspired_version_policy = std::move(aspired_version_policy);
  manager_options.num_load_threads = options_.num_load_threads;
  manager_options.num_unload_threads = options_.num_unload_threads;
  manager_options.max_num_load_retries = options_.max_num_load_retries;
  manager_options.load_retry_interval_micros =
      options_.load_retry_interval_micros;
  manager_options.pre_load_hook = std::move(options_.pre_load_hook);
  manager_options.flush_filesystem_caches = options_.flush_filesystem_caches;
  manager_options.enable_reload_servables_with_error =
      options_.enable_reload_servables_with_error;
  const tensorflow::Status status =
      AspiredVersionsManager::Create(std::move(manager_options), manager);
  if (!status.ok()) {
    VLOG(1) << "Unable to CreateAspiredVersionsManager due to: " << status;
  }
  return status;
}

Status ServerCore::CreateResourceTracker(
    std::unique_ptr<ResourceTracker>* resource_tracker) {
  ResourceUtil::Options resource_util_options;
  resource_util_options.devices[device_types::kMain] = 1;
  auto resource_util =
      std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));
  ResourceAllocation total_resources;
  resource_util->SetQuantity(
      resource_util->CreateBoundResource(device_types::kMain,
                                         resource_kinds::kRamBytes),
      options_.total_model_memory_limit_bytes, &total_resources);
  const tensorflow::Status status = ResourceTracker::Create(
      total_resources, std::move(resource_util), resource_tracker);
  if (!status.ok()) {
    VLOG(1) << "Unable to CreateResourceTracker due to: " << status;
  }
  return status;
}

// ************************************************************************
// Request Processing.
// ************************************************************************

Status ServerCore::ServableRequestFromModelSpec(
    const ModelSpec& model_spec, ServableRequest* servable_request) const {
  if (model_spec.name().empty()) {
    return errors::InvalidArgument("ModelSpec has no name specified.");
  }

  switch (model_spec.version_choice_case()) {
    case ModelSpec::kVersion: {
      *servable_request = ServableRequest::Specific(
          model_spec.name(), model_spec.version().value());
      break;
    }
    case ModelSpec::kVersionLabel: {
      if (!options_.allow_version_labels) {
        return errors::InvalidArgument(
            "ModelSpec has 'version_label' set, but it is not currently "
            "allowed by the server.");
      }
      int64_t version;
      TF_RETURN_IF_ERROR(GetModelVersionForLabel(
          model_spec.name(), model_spec.version_label(), &version));
      *servable_request = ServableRequest::Specific(model_spec.name(), version);
      break;
    }
    case ModelSpec::VERSION_CHOICE_NOT_SET: {
      *servable_request = ServableRequest::Latest(model_spec.name());
      break;
    }
  }
  return OkStatus();
}

Status ServerCore::GetModelVersionForLabel(const string& model_name,
                                           const string& label,
                                           int64_t* version) const {
  mutex_lock l(model_labels_to_versions_mu_);
  if (model_labels_to_versions_ == nullptr) {
    return errors::Unavailable(
        strings::StrCat("Model labels does not init yet.", label));
  }
  auto version_map_it = model_labels_to_versions_->find(model_name);
  if (version_map_it != model_labels_to_versions_->end()) {
    const std::map<string, int64_t>& version_map = version_map_it->second;
    auto version_it = version_map.find(label);
    if (version_it != version_map.end()) {
      *version = version_it->second;
      return OkStatus();
    }
  }
  return errors::InvalidArgument(
      strings::StrCat("Unrecognized servable version label: ", label));
}

}  //  namespace serving
}  //  namespace tensorflow
