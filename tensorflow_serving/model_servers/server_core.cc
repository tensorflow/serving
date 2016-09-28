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

#include "google/protobuf/any.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/core/eager_load_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"

namespace tensorflow {
namespace serving {

// ************************************************************************
// Local Helper Methods.
// ************************************************************************

namespace {

// Returns an error if it is not the case that all ModelConfigList models have
// the same model type, otherwise returns OK and sets 'model_type' to the type.
Status ValidateAllListedModelsAreOfSamePlatform(const ModelServerConfig& config,
                                                string* model_platform) {
  for (const auto& model : config.model_config_list().config()) {
    // Get platform (with backward compatibility)
    string platform;
    if (model.model_type() != ModelType::MODEL_TYPE_UNSPECIFIED) {
      if (!model.model_platform().empty()) {
        return errors::InvalidArgument(
            "Illegal setting both model_type(deprecated) and model_platform.");
      }
      if (model.model_type() == ModelType::TENSORFLOW) {
        platform = kTensorFlowModelPlatform;
      } else {
        platform = kOtherModelPlatform;
      }
    } else {
      platform = model.model_platform();
    }

    if (platform.empty()) {
      return errors::InvalidArgument(
          "Illegal setting neither model_type(deprecated) nor model_platform.");
    }

    // Check if matches found_platform (so far)
    if (model_platform->empty()) {
      *model_platform = platform;
    }
    // Error if not, continue if true
    if (platform != *model_platform) {
      return errors::InvalidArgument(
          "Expect all models to have the same type.");
    }
  }
  return Status::OK();
}

}  // namespace

// ************************************************************************
// Public Methods.
// ************************************************************************

Status ServerCore::Create(
    const ModelServerConfig& config,
    const SourceAdapterCreator& source_adapter_creator,
    const ServableStateMonitorCreator& servable_state_monitor_creator,
    const CustomModelConfigLoader& custom_model_config_loader,
    const ServerCoreConfig& server_core_config,
    std::unique_ptr<ServerCore>* server_core) {
  server_core->reset(
      new ServerCore(source_adapter_creator, servable_state_monitor_creator,
                     custom_model_config_loader, server_core_config));
  TF_RETURN_IF_ERROR((*server_core)->Initialize());
  return (*server_core)->ReloadConfig(config);
}

// ************************************************************************
// Server Setup and Initialization.
// ************************************************************************

ServerCore::ServerCore(
    const SourceAdapterCreator& source_adapter_creator,
    const ServableStateMonitorCreator& servable_state_monitor_creator,
    const CustomModelConfigLoader& custom_model_config_loader,
    const ServerCoreConfig& server_core_config)
    : source_adapter_creator_(source_adapter_creator),
      servable_state_monitor_creator_(servable_state_monitor_creator),
      custom_model_config_loader_(custom_model_config_loader),
      server_core_config_(server_core_config),
      servable_event_bus_(EventBus<ServableState>::CreateEventBus()) {}

Status ServerCore::Initialize() {
  std::unique_ptr<ServableStateMonitor> servable_state_monitor;
  TF_RETURN_IF_ERROR(servable_state_monitor_creator_(servable_event_bus_.get(),
                                                     &servable_state_monitor));
  servable_state_monitor_ = std::move(servable_state_monitor);

  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  TF_RETURN_IF_ERROR(CreateAspiredVersionsManager(&aspired_versions_manager));
  manager_.SetOwned(std::move(aspired_versions_manager));

  return Status::OK();
}

Status ServerCore::WaitUntilConfiguredModelsAvailable() {
  std::vector<ServableRequest> awaited_models;
  for (const auto& model : config_.model_config_list().config()) {
    awaited_models.push_back(ServableRequest::Latest(model.name()));
  }
  std::map<ServableId, ServableState::ManagerState> states_reached;
  const bool all_models_available =
      servable_state_monitor_->WaitUntilServablesReachState(
          awaited_models, ServableState::ManagerState::kAvailable,
          &states_reached);
  if (!all_models_available) {
    string message = "Some models did not become available: {";
    for (const auto& id_and_state : states_reached) {
      if (id_and_state.second != ServableState::ManagerState::kAvailable) {
        strings::StrAppend(&message, id_and_state.first.DebugString(), ", ");
      }
    }
    strings::StrAppend(&message, "}");
    return errors::Unknown(message);
  }
  return Status::OK();
}

Status ServerCore::AddModelsViaModelConfigList() {
  // Config validation.
  string model_platform;
  TF_RETURN_IF_ERROR(
      ValidateAllListedModelsAreOfSamePlatform(config_, &model_platform));

  // Create the source adapter if we haven't done so.
  bool is_first_config = storage_path_source_ == nullptr;
  ModelServerSourceAdapter* source_adapter = nullptr;
  if (is_first_config) {
    model_platform_ = model_platform;
    std::unique_ptr<ModelServerSourceAdapter> new_source_adapter;
    TF_RETURN_IF_ERROR(CreateSourceAdapter(model_platform_, manager_.get(),
                                           &new_source_adapter));
    source_adapter = new_source_adapter.get();
    manager_.AddDependency(std::move(new_source_adapter));
  }

  // Determine if config transition is legal.
  if (!is_first_config && model_platform_ != model_platform) {
    return errors::FailedPrecondition(
        "Cannot transition to requested model platform. It is only legal to "
        "transition to the same model platform.");
  }

  // Create/reload file system storage path source.
  const FileSystemStoragePathSourceConfig source_config =
      CreateStoragePathSourceConfig(config_);
  if (is_first_config) {
    TF_RETURN_IF_ERROR(
        CreateFileSystemStoragePathSource(source_config, source_adapter));
  } else {
    TF_RETURN_IF_ERROR(ReloadFileSystemStoragePathSourceConfig(source_config));
  }
  return Status::OK();
}

Status ServerCore::AddModelsViaCustomModelConfig() {
  return custom_model_config_loader_(config_.custom_model_config(),
                                     servable_event_bus_.get(), manager_.get());
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
    return Status::OK();
  }
  config_ = new_config;

  LOG(INFO) << "Adding/updating models.";
  switch (config_.config_case()) {
    case ModelServerConfig::kModelConfigList: {
      TF_RETURN_IF_ERROR(AddModelsViaModelConfigList());
      TF_RETURN_IF_ERROR(WaitUntilConfiguredModelsAvailable());
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

  return Status::OK();
}

Status ServerCore::CreateSourceAdapter(
    const string& model_platform, Target<std::unique_ptr<Loader>>* target,
    std::unique_ptr<ModelServerSourceAdapter>* adapter) {
  TF_RETURN_IF_ERROR(source_adapter_creator_(model_platform, adapter));
  ConnectSourceToTarget(adapter->get(), target);
  return Status::OK();
}

FileSystemStoragePathSourceConfig ServerCore::CreateStoragePathSourceConfig(
    const ModelServerConfig& config) const {
  FileSystemStoragePathSourceConfig source_config;
  source_config.set_file_system_poll_wait_seconds(
      server_core_config_.file_system_poll_wait_seconds);
  for (const auto& model : config.model_config_list().config()) {
    LOG(INFO) << " (Re-)adding model: " << model.name();
    FileSystemStoragePathSourceConfig::ServableToMonitor* servable =
        source_config.add_servables();
    servable->set_servable_name(model.name());
    servable->set_base_path(model.base_path());
  }
  return source_config;
}

Status ServerCore::CreateFileSystemStoragePathSource(
    const FileSystemStoragePathSourceConfig& source_config,
    Target<StoragePath>* target) {
  std::unique_ptr<FileSystemStoragePathSource> storage_path_source;
  TF_RETURN_IF_ERROR(
      FileSystemStoragePathSource::Create(source_config, &storage_path_source));
  ConnectSourceToTarget(storage_path_source.get(), target);
  storage_path_source_ = storage_path_source.get();
  manager_.AddDependency(std::move(storage_path_source));
  return Status::OK();
}

Status ServerCore::ReloadFileSystemStoragePathSourceConfig(
    const FileSystemStoragePathSourceConfig& source_config) {
  return storage_path_source_->UpdateConfig(source_config);
}

Status ServerCore::CreateAspiredVersionsManager(
    std::unique_ptr<AspiredVersionsManager>* const manager) {
  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  AspiredVersionsManager::Options manager_options;
  manager_options.servable_event_bus = servable_event_bus_.get();
  manager_options.aspired_version_policy.reset(new EagerLoadPolicy());
  manager_options.num_load_unload_threads =
      server_core_config_.num_load_unload_threads;
  manager_options.max_num_load_retries =
      server_core_config_.max_num_load_retries;
  return AspiredVersionsManager::Create(std::move(manager_options), manager);
}

// ************************************************************************
// Request Processing.
// ************************************************************************

Status ServerCore::ServableRequestFromModelSpec(
    const ModelSpec& model_spec, ServableRequest* servable_request) const {
  if (model_spec.name().empty()) {
    return errors::InvalidArgument("ModelSpec has no name specified.");
  }
  if (model_spec.has_version()) {
    *servable_request = ServableRequest::Specific(model_spec.name(),
                                                  model_spec.version().value());
  } else {
    *servable_request = ServableRequest::Latest(model_spec.name());
  }
  return Status::OK();
}

// ************************************************************************
// Test Access.
// ************************************************************************

std::vector<ServableId> ServerCore::ListAvailableServableIds() const {
  return manager_->ListAvailableServableIds();
}

}  //  namespace serving
}  //  namespace tensorflow
