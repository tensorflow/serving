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

#include "tensorflow_serving/core/caching_manager.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

Status CachingManager::Create(
    Options options, std::unique_ptr<LoaderFactory> loader_factory,
    std::unique_ptr<CachingManager>* caching_manager) {
  // Set up basic manager options from the caching manager options.
  BasicManager::Options basic_manager_options;
  basic_manager_options.resource_tracker = std::move(options.resource_tracker);
  basic_manager_options.num_load_threads = options.num_load_threads;
  basic_manager_options.num_unload_threads = options.num_unload_threads;
  basic_manager_options.max_num_load_retries = options.max_num_load_retries;
  basic_manager_options.load_retry_interval_micros =
      options.load_retry_interval_micros;
  basic_manager_options.servable_event_bus = options.servable_event_bus;

  // Create a basic manager and use it to construct the caching manager.
  std::unique_ptr<BasicManager> basic_manager;
  TF_RETURN_IF_ERROR(
      BasicManager::Create(std::move(basic_manager_options), &basic_manager));

  caching_manager->reset(
      new CachingManager(std::move(loader_factory), std::move(basic_manager)));
  return Status::OK();
}

CachingManager::CachingManager(std::unique_ptr<LoaderFactory> loader_factory,
                               std::unique_ptr<BasicManager> basic_manager)
    : loader_factory_(std::move(loader_factory)),
      basic_manager_(std::move(basic_manager)) {}

CachingManager::~CachingManager() {}

Status CachingManager::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const handle) {
  if (request.version) {
    return GetUntypedServableHandleForId({request.name, *request.version},
                                         handle);
  }
  // Since there is no explicit version in the request, get the latest from the
  // loader-factory.
  const int64 latest_version = loader_factory_->GetLatestVersion(request.name);
  return GetUntypedServableHandleForId({request.name, latest_version}, handle);
}

Status CachingManager::GetUntypedServableHandleForId(
    const ServableId& servable_id,
    std::unique_ptr<UntypedServableHandle>* handle) {
  // Check if the underlying basic manager can already serve this request.
  const Status handle_status = basic_manager_->GetUntypedServableHandle(
      ServableRequest::FromId(servable_id), handle);

  // If the servable is already managed and loaded by the basic manager, serve
  // it.
  if (handle_status.ok() || handle_status.code() != error::NOT_FOUND) {
    return handle_status;
  }

  // Build the servable data corresponding to the servable-id.
  ServableData<std::unique_ptr<Loader>> loader_data =
      loader_factory_->CreateLoader(servable_id);

  // Load the servable corresponding to the servable-id. For multiple concurrent
  // requests enforces that exactly one thread performs the load operation with
  // the wrapped basic-manager. All other requests block until the load
  // completes and then trivially succeed.
  TF_RETURN_IF_ERROR(LoadServable(std::move(loader_data)));

  // Return the handle using the loaded servable data now.
  return basic_manager_->GetUntypedServableHandle(
      ServableRequest::FromId(servable_id), handle);
}

Status CachingManager::LoadServable(
    ServableData<std::unique_ptr<Loader>> loader_data) {
  const ServableId servable_id = loader_data.id();

  std::shared_ptr<mutex> servable_id_mu;
  {
    mutex_lock l(load_mutex_map_mu_);
    auto iter = load_mutex_map_.find(servable_id);
    if (iter == load_mutex_map_.end()) {
      iter =
          load_mutex_map_.emplace(servable_id, std::make_shared<mutex>()).first;
    }
    servable_id_mu = iter->second;
  }

  {
    // Ensure only one thread attempts to load the servable at a time.
    mutex_lock l(*servable_id_mu);

    // Retrieve the state of the servable from the wrapped basic-manager. The
    // servable should already be managed by the basic-manager.
    const optional<ServableStateSnapshot<>> snapshot =
        basic_manager_->GetManagedServableStateSnapshot(servable_id);
    if (snapshot) {
      // The servable is already being managed by 'basic_manager_'. Hence it
      // ought to be loaded, based on CachingManager's implementation invariant
      // of doing manage+load atomically.
      if (snapshot.value().state != LoaderHarness::State::kReady) {
        const string error_msg = strings::StrCat(
            "Servable requested for load is already being managed, but is not "
            "loaded: ",
            servable_id.DebugString());
        DCHECK(false) << error_msg;
        return errors::Internal(error_msg);
      }
    } else {
      // Load the servable since it has not been loaded yet based on its state.
      //
      // First, transfer the servable to the basic manager. The loader_data may
      // contain an error and the basic manager is equipped to handle that
      // appropriately. By propagating such errors back to the basic manager,
      // the functionality of the event-bus and the servable state monitor are
      // automatically available in the caching-manager as well (via the basic
      // manager).
      const Status manage_status =
          basic_manager_->ManageServable(std::move(loader_data));
      if (!manage_status.ok()) {
        const string error_msg = strings::StrCat(
            "Internal error: unable to transfer servable to 'basic_manager_': ",
            manage_status.error_message());
        DCHECK(false) << error_msg;
        return errors::Internal(error_msg);
      }

      Notification load_done;
      Status load_status;
      basic_manager_->LoadServable(servable_id, [&](const Status& status) {
        load_status = status;
        load_done.Notify();
      });
      load_done.WaitForNotification();
      TF_RETURN_IF_ERROR(load_status);
    }
  }
  servable_id_mu.reset();
  MaybeEraseLoadMutexMapEntry(servable_id);
  return Status::OK();
}

void CachingManager::MaybeEraseLoadMutexMapEntry(
    const ServableId& servable_id) {
  mutex_lock l(load_mutex_map_mu_);
  auto iter = load_mutex_map_.find(servable_id);
  // Erase the entry from the map if one exists and if the mutex shared_ptr
  // is the last remaining one.
  if (iter != load_mutex_map_.end() && iter->second.unique()) {
    load_mutex_map_.erase(iter);
  }
}

int64 CachingManager::GetLoadMutexMapSize() const {
  mutex_lock l(load_mutex_map_mu_);
  return load_mutex_map_.size();
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
CachingManager::GetAvailableUntypedServableHandles() const {
  return basic_manager_->GetAvailableUntypedServableHandles();
}

std::vector<ServableId> CachingManager::ListAvailableServableIds() const {
  return basic_manager_->ListAvailableServableIds();
}

PathPrefixLoaderFactory::PathPrefixLoaderFactory(
    const string& path_prefix,
    std::unique_ptr<StoragePathSourceAdapter> adapter)
    : path_prefix_(path_prefix), adapter_(std::move(adapter)) {}

ServableData<std::unique_ptr<Loader>> PathPrefixLoaderFactory::CreateLoader(
    const ServableId& id) {
  if (id.version != 0) {
    return ServableData<std::unique_ptr<Loader>>(
        id,
        errors::FailedPrecondition("PathPrefixLoaderFactory only supports "
                                   "single-version servables at version 0"));
  }
  const StoragePath servable_path = io::JoinPath(path_prefix_, id.name);
  return adapter_->AdaptOneVersion({id, servable_path});
}

int64 PathPrefixLoaderFactory::GetLatestVersion(
    const string& servable_name) const {
  return 0;
}

}  // namespace serving
}  // namespace tensorflow
