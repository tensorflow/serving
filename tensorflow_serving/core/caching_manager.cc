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
  basic_manager_options.num_load_unload_threads =
      options.num_load_unload_threads;
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
    std::unique_ptr<UntypedServableHandle>* handle) {
  // Check if the underlying basic manager can already serve this request.
  const Status handle_status =
      basic_manager_->GetUntypedServableHandle(request, handle);

  // If the servable is already managed and loaded by the basic manager, serve
  // it.
  if (handle_status.ok() || handle_status.code() != error::NOT_FOUND) {
    return handle_status;
  }

  // Build the servable data corresponding to the request.
  std::unique_ptr<ServableData<std::unique_ptr<Loader>>> loader_data;
  TF_RETURN_IF_ERROR(
      loader_factory_->CreateLoader(std::move(request), &loader_data));

  // Keep track of the servable id to use for loading the servable.
  const ServableId id = loader_data->id();

  // Manage the servable using the basic manager. The loader_data may contain an
  // error and the basic manager is equipped to handle that appropriately. By
  // propagating such errors back to the basic manager, the functionality of the
  // event-bus and the servable state monitor are automatically available in the
  // caching-manager as well (via the basic manager).
  basic_manager_->ManageServable(std::move(*loader_data));

  // Load the servable and use a notification to wait until it is complete.
  Notification load_done;
  Status load_status;
  basic_manager_->LoadServable(id, [&](const Status& status) {
    load_status = status;
    load_done.Notify();
  });
  load_done.WaitForNotification();
  TF_RETURN_IF_ERROR(load_status);

  // Return the handle using the loaded servable data now.
  return basic_manager_->GetUntypedServableHandle(request, handle);
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
CachingManager::GetAvailableUntypedServableHandles() const {
  return basic_manager_->GetAvailableUntypedServableHandles();
}

std::vector<ServableId> CachingManager::ListAvailableServableIds() const {
  return basic_manager_->ListAvailableServableIds();
}

}  // namespace serving
}  // namespace tensorflow
