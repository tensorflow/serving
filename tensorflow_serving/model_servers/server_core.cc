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
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"

namespace tensorflow {
namespace serving {

// ************************************************************************
// Public Methods.
// ************************************************************************

Status ServerCore::Create(
    const ModelServerConfig& config,
    SourceAdapterCreator source_adapter_creator,
    ServableStateMonitorCreator servable_state_monitor_creator,
    DynamicModelConfigLoader dynamic_state_config_loader,
    std::unique_ptr<ServerCore>* server_core) {
  server_core->reset(new ServerCore(source_adapter_creator,
                                    servable_state_monitor_creator,
                                    dynamic_state_config_loader));
  TF_RETURN_IF_ERROR((*server_core)->Initialize());
  return (*server_core)->ReloadConfig(config);
}

// ************************************************************************
// Server Setup and Initialization.
// ************************************************************************

ServerCore::ServerCore(
    SourceAdapterCreator source_adapter_creator,
    ServableStateMonitorCreator servable_state_monitor_creator,
    DynamicModelConfigLoader dynamic_state_config_loader)
    : source_adapter_creator_(source_adapter_creator),
      servable_state_monitor_creator_(servable_state_monitor_creator),
      dynamic_model_config_loader_(dynamic_state_config_loader),
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

Status ServerCore::AddModelsViaModelConfigList(
    const ModelServerConfig& config) {
  std::vector<ServableRequest> awaited_models;
  for (const auto& model : config.model_config_list().config()) {
    LOG(INFO) << " Adding model: " << model.name();
    awaited_models.push_back(ServableRequest::Latest(model.name()));
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>
        source_adapter;
    TF_RETURN_IF_ERROR(CreateSourceAdapter(model.model_type(), manager_.get(),
                                           &source_adapter));
    std::unique_ptr<Source<StoragePath>> path_source;
    TF_RETURN_IF_ERROR(CreateStoragePathSource(
        model.base_path(), model.name(), source_adapter.get(), &path_source));
    manager_.AddDependency(std::move(source_adapter));
    manager_.AddDependency(std::move(path_source));
  }
  std::map<ServableId, ServableState::ManagerState> states_reached;
  const bool all_configured_models_available =
      servable_state_monitor_->WaitUntilServablesReachState(
          awaited_models, ServableState::ManagerState::kAvailable,
          &states_reached);
  if (!all_configured_models_available) {
    string message = "Some models did not become available: {";
    for (const auto& id_and_state : states_reached) {
      if (id_and_state.second != ServableState::ManagerState::kAvailable) {
        strings::StrAppend(&message, id_and_state.first.DebugString(), ", ");
      }
    }
    strings::StrAppend(&message, "}");
    return Status(error::UNKNOWN, message);
  }
  return Status::OK();
}

Status ServerCore::AddModelsViaDynamicModelConfig(
    const ModelServerConfig& config) {
  return dynamic_model_config_loader_(config.dynamic_model_config(),
                                      manager_.get());
}

Status ServerCore::ReloadConfig(const ModelServerConfig& config) {
  {
    mutex_lock m(seen_models_mu_);
    if (seen_models_) {
      return errors::Internal("Repeated ReloadConfig calls not supported.");
    }
    if (config.config_case() == ModelServerConfig::CONFIG_NOT_SET) {
      // Nothing to load. In this case we allow a future call with a non-empty
      // config.
      LOG(INFO) << "Taking no action for empty config.  Future Reloads "
                << "are allowed.";
      return Status::OK();
    }
    seen_models_ = true;
  }
  LOG(INFO) << "Adding models to manager.";
  switch (config.config_case()) {
    case ModelServerConfig::kModelConfigList:
      TF_RETURN_IF_ERROR(AddModelsViaModelConfigList(config));
      break;
    case ModelServerConfig::kDynamicModelConfig:
      TF_RETURN_IF_ERROR(AddModelsViaDynamicModelConfig(config));
      break;
    default:
      return errors::InvalidArgument("Invalid ServerModelConfig");
  }

  return Status::OK();
}

Status ServerCore::CreateSourceAdapter(
    const string& model_type, Target<std::unique_ptr<Loader>>* target,
    std::unique_ptr<ModelServerSourceAdapter>* adapter) {
  TF_RETURN_IF_ERROR(source_adapter_creator_(model_type, adapter));
  ConnectSourceToTarget(adapter->get(), target);
  return Status::OK();
}

Status ServerCore::CreateStoragePathSource(
    const string& base_path, const string& servable_name,
    Target<StoragePath>* target,
    std::unique_ptr<Source<StoragePath>>* path_source) {
  FileSystemStoragePathSourceConfig config;
  config.set_servable_name(servable_name);
  config.set_base_path(base_path);
  config.set_file_system_poll_wait_seconds(30);

  std::unique_ptr<FileSystemStoragePathSource> file_system_source;
  TF_RETURN_IF_ERROR(
      FileSystemStoragePathSource::Create(config, &file_system_source));
  ConnectSourceToTarget(file_system_source.get(), target);
  *path_source = std::move(file_system_source);
  return Status::OK();
}

Status ServerCore::CreateAspiredVersionsManager(
    std::unique_ptr<AspiredVersionsManager>* const manager) {
  AspiredVersionsManager::Options manager_options;
  manager_options.servable_event_bus = servable_event_bus_.get();
  manager_options.aspired_version_policy.reset(new EagerLoadPolicy());
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
