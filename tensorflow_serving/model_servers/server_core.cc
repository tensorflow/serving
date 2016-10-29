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
#include "tensorflow_serving/core/load_servables_fast.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
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

Status ServerCore::Create(Options options,
                          std::unique_ptr<ServerCore>* server_core) {
  if (options.source_adapter_creator == nullptr) {
    options.source_adapter_creator = [](
        const string& model_platform,
        std::unique_ptr<ServerCore::ModelServerSourceAdapter>* adapter) {
      SessionBundleSourceAdapterConfig source_adapter_config;
      if (model_platform != kTensorFlowModelPlatform) {
        return errors::InvalidArgument(
            "ModelServer supports only TensorFlow model.");
      }
      std::unique_ptr<SessionBundleSourceAdapter> typed_adapter;
      TF_RETURN_IF_ERROR(SessionBundleSourceAdapter::Create(
          source_adapter_config, &typed_adapter));
      *adapter = std::move(typed_adapter);
      return Status::OK();
    };
  }

  if (options.servable_state_monitor_creator == nullptr) {
    options.servable_state_monitor_creator = [](
        EventBus<ServableState>* event_bus,
        std::unique_ptr<ServableStateMonitor>* monitor) {
      monitor->reset(new ServableStateMonitor(event_bus));
      return Status::OK();
    };
  }

  // We need to move the aspired_version_policy first because we will move the
  // server_core_config (which contains aspired_version_policy) below.
  std::unique_ptr<AspiredVersionPolicy> aspired_version_policy =
      std::move(options.aspired_version_policy);
  server_core->reset(new ServerCore(std::move(options)));
  TF_RETURN_IF_ERROR(
      (*server_core)->Initialize(std::move(aspired_version_policy)));
  return (*server_core)->ReloadConfig(options.model_server_config);
}

// ************************************************************************
// Server Setup and Initialization.
// ************************************************************************

ServerCore::ServerCore(Options options)
    : options_(std::move(options)),
      servable_event_bus_(EventBus<ServableState>::CreateEventBus()) {}

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
  const bool is_first_config = storage_path_source_ == nullptr;
  if (is_first_config) {
    model_platform_ = model_platform;
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
    std::unique_ptr<ModelServerSourceAdapter> source_adapter;
    TF_RETURN_IF_ERROR(CreateSourceAdapter(model_platform_, &source_adapter));
    TF_RETURN_IF_ERROR(
        CreateFileSystemStoragePathSource(source_config, source_adapter.get()));
    std::vector<ServableRequest> static_servables;
    for (const auto& model : config_.model_config_list().config()) {
      static_servables.push_back(ServableRequest::Latest(model.name()));
    }
    const tensorflow::Status status = ConnectSourceWithFastInitialLoad(
        manager_.get(), source_adapter.get(), servable_state_monitor_.get(),
        static_servables, options_.num_initial_load_unload_threads);
    if (!status.ok()) {
      VLOG(1) << "Unable to ConnectSourceWithFastInitialLoad due to: "
              << status;
      return status;
    }
    manager_.AddDependency(std::move(source_adapter));
  } else {
    TF_RETURN_IF_ERROR(ReloadFileSystemStoragePathSourceConfig(source_config));
    TF_RETURN_IF_ERROR(WaitUntilConfiguredModelsAvailable());
  }
  return Status::OK();
}

Status ServerCore::AddModelsViaCustomModelConfig() {
  if (options_.custom_model_config_loader == nullptr) {
    return errors::InvalidArgument(
        "Missing custom_model_config_loader in ServerCore Options");
  }

  return options_.custom_model_config_loader(
      config_.custom_model_config(), servable_event_bus_.get(), &manager_);
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
    const string& model_platform,
    std::unique_ptr<ModelServerSourceAdapter>* adapter) {
  const tensorflow::Status status =
      options_.source_adapter_creator(model_platform, adapter);
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
  for (const auto& model : config.model_config_list().config()) {
    LOG(INFO) << " (Re-)adding model: " << model.name();
    FileSystemStoragePathSourceConfig::ServableToMonitor* servable =
        source_config.add_servables();
    servable->set_servable_name(model.name());
    servable->set_base_path(model.base_path());
    servable->set_version_policy(model.version_policy());
  }
  return source_config;
}

Status ServerCore::CreateFileSystemStoragePathSource(
    const FileSystemStoragePathSourceConfig& source_config,
    Target<StoragePath>* target) {
  std::unique_ptr<FileSystemStoragePathSource> storage_path_source;
  const tensorflow::Status status =
      FileSystemStoragePathSource::Create(source_config, &storage_path_source);
  if (!status.ok()) {
    VLOG(1) << "Unable to create FileSystemStoragePathSource due to: "
            << status;
    return status;
  }
  ConnectSourceToTarget(storage_path_source.get(), target);
  storage_path_source_ = storage_path_source.get();
  manager_.AddDependency(std::move(storage_path_source));
  return Status::OK();
}

Status ServerCore::ReloadFileSystemStoragePathSourceConfig(
    const FileSystemStoragePathSourceConfig& source_config) {
  const tensorflow::Status status =
      storage_path_source_->UpdateConfig(source_config);
  if (!status.ok()) {
    VLOG(1) << "Unable to ReloadFileSystemStoragePathSourceConfig due to: "
            << status;
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
  manager_options.num_load_unload_threads = options_.num_load_unload_threads;
  manager_options.max_num_load_retries = options_.max_num_load_retries;
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
  ResourceAllocation::Entry* main_memory_resource =
      total_resources.add_resource_quantities();
  main_memory_resource->mutable_resource()->set_device(device_types::kMain);
  main_memory_resource->mutable_resource()->set_kind(resource_kinds::kRamBytes);
  main_memory_resource->set_quantity(options_.total_model_memory_limit_bytes);
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
  if (model_spec.has_version()) {
    *servable_request = ServableRequest::Specific(model_spec.name(),
                                                  model_spec.version().value());
  } else {
    *servable_request = ServableRequest::Latest(model_spec.name());
  }
  return Status::OK();
}

}  //  namespace serving
}  //  namespace tensorflow
