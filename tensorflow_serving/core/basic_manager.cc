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

#include "tensorflow_serving/core/basic_manager.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/util/cleanup.h"
#include "tensorflow_serving/util/hash.h"
#include "tensorflow_serving/util/inline_executor.h"
#include "tensorflow_serving/util/threadpool_executor.h"

namespace tensorflow {
namespace serving {

struct BasicManager::ServingMap::EqRequest {
  bool operator()(const ServableRequest& lhs,
                  const ServableRequest& rhs) const {
    if (lhs.version != rhs.version) {
      return false;
    }
    // Even if there is a small probability that version checking can eliminate
    // string checking, we should do that since O(string_equality) >>
    // O(version_equality)
    if (lhs.name != rhs.name) {
      return false;
    }
    return true;
  }
};

struct BasicManager::ServingMap::HashRequest {
  uint64 operator()(const ServableRequest& request) const {
    // Hash codes for many common types are remarkably bad, often clustering
    // around the same values of the low and/or high bits for linear
    // sequences of inputs such as 1, 2, 3; or addresses of consecutively
    // allocated objects.  For these cases the default hash function is the
    // identity function on the bit patterns.
    //
    // So we apply a one-to-one mapping to the resulting bit patterns to
    // make the high bits contain more entropy from the entire hash code.
    // It's based on Fibonacci hashing from Knuth's Art of Computer
    // Programming volume 3, section 6.4.
    const uint64 version_hash = [&]() -> uint64 {
      if (request.version) {
        return std::hash<int64>()(request.version.value()) *
               0x9E3779B97F4A7C13;  // (sqrt(5) - 1)/2 as a binary fraction.
      } else {
        return 0xDECAFCAFFE;
      }
    }();
    // Using version_hash as the seed here to combine the hashes.
    return HashCombine(version_hash, std::hash<string>()(request.name));
  }
};

BasicManager::ServingMap::ServingMap()
    : handles_map_(std::unique_ptr<HandlesMap>(new HandlesMap())) {}

std::vector<ServableId> BasicManager::ServingMap::ListAvailableServableIds()
    const {
  std::vector<ServableId> ids;
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  for (auto iter = handles_map->begin(); iter != handles_map->end();) {
    // We get the iterator where all the values for a particular key ends.
    const auto key_end = handles_map->equal_range(iter->first).second;

    for (; iter != key_end; ++iter) {
      if (iter->first.version) {
        ids.push_back(iter->second->id());
      }
    }
  }
  return ids;
}

Status BasicManager::ServingMap::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  const auto found_it = handles_map->find(request);
  if (found_it == handles_map->end()) {
    return errors::NotFound("Servable not found for request: ",
                            request.DebugString());
  }

  const LoaderHarness& harness = *found_it->second;
  // We use the aliasing constructor of shared_ptr here. So even though we are
  // returning a shared_ptr to servable, the ref-counting is happening on the
  // handles_map. This delays the map destruction till the last handle from the
  // previous map is freed, when we are doing handles_map updates.
  untyped_handle->reset(new SharedPtrHandle(
      harness.id(), std::shared_ptr<Loader>(handles_map, harness.loader())));
  return Status::OK();
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
BasicManager::ServingMap::GetAvailableUntypedServableHandles() const {
  std::map<ServableId, std::unique_ptr<UntypedServableHandle>> result;
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  for (const auto& handle : *handles_map) {
    const ServableRequest& request = handle.first;
    // If the entry is the one for the latest request, skip it. We would already
    // get it from the entry which has the specific request.
    if (!request.version) {
      continue;
    }
    const LoaderHarness& harness = *handle.second;
    result.emplace(harness.id(),
                   std::unique_ptr<UntypedServableHandle>(new SharedPtrHandle(
                       harness.id(), std::shared_ptr<Loader>(
                                         handles_map, harness.loader()))));
  }
  return result;
}

void BasicManager::ServingMap::Update(const ManagedMap& managed_map) {
  struct CompareRequests {
    bool operator()(const ServableRequest& lhs,
                    const ServableRequest& rhs) const {
      const int strcmp_result = lhs.name.compare(rhs.name);
      if (strcmp_result != 0) {
        return strcmp_result < 0;
      }
      DCHECK(lhs.version);
      DCHECK(rhs.version);
      return lhs.version.value() < rhs.version.value();
    }
  };
  std::multimap<ServableRequest, std::shared_ptr<const LoaderHarness>,
                CompareRequests>
      sorted_available_map;
  for (const auto& elem : managed_map) {
    std::shared_ptr<const LoaderHarness> harness = elem.second;
    if (harness->state() == LoaderHarness::State::kReady) {
      sorted_available_map.emplace(ServableRequest::FromId(harness->id()),
                                   harness);
    }
  }

  std::unique_ptr<HandlesMap> new_handles_map(new HandlesMap());
  for (auto iter = sorted_available_map.begin();
       iter != sorted_available_map.end(); ++iter) {
    std::shared_ptr<const LoaderHarness> harness = iter->second;
    new_handles_map->emplace(ServableRequest::FromId(harness->id()), harness);
    // If this is the last harness in the stream, add it again to the
    // handles_map, marking it as the latest for that stream.
    const auto next_iter = std::next(iter);
    if (next_iter == sorted_available_map.end() ||
        next_iter->second->id().name != harness->id().name) {
      const ServableRequest latest_request =
          ServableRequest::Latest(harness->id().name);
      new_handles_map->emplace(latest_request, harness);
    }
  }

  // This blocks until the last handle given out by the old handles map is
  // freed.
  handles_map_.Update(std::move(new_handles_map));
}

Status BasicManager::Create(Options options,
                            std::unique_ptr<BasicManager>* manager) {
  std::unique_ptr<Executor> load_unload_executor;
  if (options.num_load_unload_threads == 0) {
    LOG(INFO) << "Using InlineExecutor for BasicManager.";
    load_unload_executor.reset(new InlineExecutor());
  } else {
    LOG(INFO) << "Using ThreadPoolExecutor for BasicManager with "
                 "num_load_unload_threads: "
              << options.num_load_unload_threads;
    load_unload_executor.reset(new ThreadPoolExecutor(
        options.env, "BasicManager_LoadUnload_ThreadPool",
        options.num_load_unload_threads));
  }

  LoaderHarness::Options harness_options;
  harness_options.max_num_load_retries = options.max_num_load_retries;
  harness_options.load_retry_interval_micros =
      options.load_retry_interval_micros;
  manager->reset(new BasicManager(std::move(load_unload_executor),
                                  std::move(options.resource_tracker),
                                  options.servable_event_bus, harness_options));
  return Status::OK();
}

BasicManager::BasicManager(std::unique_ptr<Executor> load_unload_executor,
                           std::unique_ptr<ResourceTracker> resource_tracker,
                           EventBus<ServableState>* servable_event_bus,
                           const LoaderHarness::Options& harness_options)
    : harness_options_(harness_options),
      servable_event_bus_(servable_event_bus) {
  load_unload_executor_ = std::move(load_unload_executor);
  resource_tracker_ = std::move(resource_tracker);
}

BasicManager::~BasicManager() {
  // Reset the executor first to finish all pending loads/unloads.
  load_unload_executor_.reset();
  UnloadAllServables();
}

void BasicManager::UnloadAllServables() {
  LOG(INFO) << "Unload all remaining servables in the manager.";
  {
    mutex_lock l(mu_);
    for (auto it = managed_map_.begin(); it != managed_map_.end(); ++it) {
      LoaderHarness* const harness = it->second.get();
      if (harness->state() == LoaderHarness::State::kReady) {
        harness->UnloadRequested();
        harness->StartQuiescing();
        harness->DoneQuiescing();
        harness->Unload();
      }
      if (harness->state() == LoaderHarness::State::kQuiescing) {
        harness->DoneQuiescing();
        harness->Unload();
      }
      if (harness->state() == LoaderHarness::State::kQuiesced) {
        harness->Unload();
      }
    }
  }
}

std::vector<ServableId> BasicManager::ListAvailableServableIds() const {
  return serving_map_.ListAvailableServableIds();
}

Status BasicManager::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  return serving_map_.GetUntypedServableHandle(request, untyped_handle);
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
BasicManager::GetAvailableUntypedServableHandles() const {
  return serving_map_.GetAvailableUntypedServableHandles();
}

void BasicManager::UpdateServingMap() {
  // This blocks until the last handle given out by the old serving map is
  // freed.
  serving_map_.Update(managed_map_);
}

BasicManager::ManagedMap::iterator BasicManager::FindHarnessInMap(
    const ServableId& id) {
  const auto range = managed_map_.equal_range(id.name);
  for (auto iter = range.first; iter != range.second; ++iter) {
    if (iter->second->id().version == id.version) {
      return iter;
    }
  }
  return managed_map_.end();
}

void BasicManager::DeleteHarness(const ServableId& id) {
  const auto it = FindHarnessInMap(id);
  DCHECK(it != managed_map_.end());
  if (it == managed_map_.end()) {
    LOG(ERROR) << "Request to delete harness for " << id
               << ", but no such harness found in managed_map_";
    return;
  }
  managed_map_.erase(it);
}

Status BasicManager::ManageServableInternal(
    ServableData<std::unique_ptr<Loader>> servable,
    std::function<std::shared_ptr<LoaderHarness>(const ServableId&,
                                                 std::unique_ptr<Loader>)>
        harness_creator) {
  mutex_lock l(mu_);

  const auto iter = BasicManager::FindHarnessInMap(servable.id());
  if (iter != managed_map_.end()) {
    return errors::FailedPrecondition(
        "This servable is already being managed: ",
        servable.id().DebugString());
  }

  std::unique_ptr<Loader> loader;
  if (servable.status().ok()) {
    loader = servable.ConsumeDataOrDie();
  }

  std::shared_ptr<LoaderHarness> harness =
      harness_creator(servable.id(), std::move(loader));
  if (!servable.status().ok()) {
    harness->Error(servable.status());
    PublishOnEventBus(
        {harness->id(), ServableState::ManagerState::kEnd, harness->status()});
  } else {
    PublishOnEventBus({harness->id(), ServableState::ManagerState::kStart,
                       harness->status()});
    managed_map_.emplace(servable.id().name, harness);
  }

  return Status::OK();
}

Status BasicManager::ManageServable(
    ServableData<std::unique_ptr<Loader>> servable) {
  return ManageServableInternal(
      std::move(servable),
      [this](const ServableId& id, std::unique_ptr<Loader> loader) {
        return std::make_shared<LoaderHarness>(id, std::move(loader),
                                               harness_options_);
      });
}

Status BasicManager::GetHealthyHarness(const ServableId& id,
                                       LoaderHarness** harness) {
  // Look up the request servable's harness.
  auto iter = FindHarnessInMap(id);
  if (iter == managed_map_.end()) {
    return errors::NotFound(
        "This servable is not being managed by the manager: ",
        id.DebugString());
  }
  TF_RETURN_IF_ERROR(iter->second->status());
  *harness = iter->second.get();
  return Status::OK();
}

std::vector<const Loader*> BasicManager::GetLoadersCurrentlyUsingResources()
    const {
  std::vector<const Loader*> loaders;
  for (const auto& entry : managed_map_) {
    const LoaderHarness& harness = *entry.second;
    bool uses_resources;
    switch (harness.state()) {
      case LoaderHarness::State::kNew:
        uses_resources = false;
        break;
      case LoaderHarness::State::kLoadRequested:
        uses_resources = false;
        break;
      case LoaderHarness::State::kLoadApproved:
        uses_resources = true;
        break;
      case LoaderHarness::State::kLoading:
        uses_resources = true;
        break;
      case LoaderHarness::State::kReady:
        uses_resources = true;
        break;
      case LoaderHarness::State::kQuiescing:
        uses_resources = true;
        break;
      case LoaderHarness::State::kQuiesced:
        uses_resources = true;
        break;
      case LoaderHarness::State::kUnloadRequested:
        uses_resources = true;
        break;
      case LoaderHarness::State::kUnloading:
        uses_resources = true;
        break;
      case LoaderHarness::State::kDisabled:
        uses_resources = false;
        break;
      case LoaderHarness::State::kError:
        uses_resources = false;
        break;
    }
    if (uses_resources) {
      loaders.push_back(harness.loader());
    }
  }
  return loaders;
}

std::vector<string> BasicManager::GetManagedServableNames() const {
  mutex_lock l(mu_);

  std::vector<string> servable_names;
  for (auto iter = managed_map_.begin(); iter != managed_map_.end();
       iter = managed_map_.equal_range(iter->first).second) {
    servable_names.push_back(iter->first);
  }
  return servable_names;
}

Status BasicManager::ExecuteLoad(LoaderHarness* harness) {
  PublishOnEventBus({harness->id(), ServableState::ManagerState::kLoading,
                     harness->status()});
  const auto load_status = harness->Load(ResourceAllocation());

  {
    mutex_lock l(mu_);

    if (!load_status.ok()) {
      PublishOnEventBus({harness->id(), ServableState::ManagerState::kEnd,
                         harness->status()});
      DeleteHarness(harness->id());
      return load_status;
    }

    UpdateServingMap();
    PublishOnEventBus({harness->id(), ServableState::ManagerState::kAvailable,
                       harness->status()});
  }
  return Status::OK();
}

void BasicManager::LoadServable(const ServableId& id,
                                const DoneCallback done_callback) {
  LoadOrUnloadRequest request;
  request.kind = LoadOrUnloadRequest::Kind::kLoad;
  request.servable_id = id;
  LoadOrUnloadServable(request, done_callback);
}

void BasicManager::CancelLoadServableRetry(const ServableId& id) {
  mutex_lock l(mu_);
  LoaderHarness* harness;
  const Status status = GetHealthyHarness(id, &harness);
  if (!status.ok()) {
    return;
  }
  harness->set_cancel_load_retry(true);
}

Status BasicManager::ExecuteUnload(LoaderHarness* harness) {
  {
    // StartQuiescing() would have been already called.
    mutex_lock l(mu_);
    PublishOnEventBus({harness->id(), ServableState::ManagerState::kUnloading,
                       harness->status()});
    UpdateServingMap();
    harness->DoneQuiescing();
  }

  harness->Unload();

  {
    mutex_lock l(mu_);
    auto iter = FindHarnessInMap(harness->id());
    PublishOnEventBus({iter->second->id(), ServableState::ManagerState::kEnd,
                       iter->second->status()});
    // This erase will lead to the LoaderHarness being deleted.
    managed_map_.erase(iter);
  }
  return Status::OK();
}

void BasicManager::UnloadServable(const ServableId& id,
                                  const DoneCallback done_callback) {
  LoadOrUnloadRequest request;
  request.kind = LoadOrUnloadRequest::Kind::kUnload;
  request.servable_id = id;
  LoadOrUnloadServable(request, done_callback);
}

Status BasicManager::ExecuteLoadOrUnload(const LoadOrUnloadRequest& request,
                                         LoaderHarness* harness) {
  Status execution_status;
  switch (request.kind) {
    case LoadOrUnloadRequest::Kind::kLoad:
      execution_status = ExecuteLoad(harness);
      break;
    case LoadOrUnloadRequest::Kind::kUnload:
      execution_status = ExecuteUnload(harness);
      break;
  }

  {
    mutex_lock l(mu_);
    --num_ongoing_load_unload_executions_;
    DCHECK_GE(num_ongoing_load_unload_executions_, 0);
    num_ongoing_load_unload_executions_cv_.notify_all();
  }

  return execution_status;
}

void BasicManager::LoadOrUnloadServable(const LoadOrUnloadRequest& request,
                                        DoneCallback done_callback) {
  const Status status = [&]() {
    mutex_lock l(mu_);
    LoaderHarness* harness;
    TF_RETURN_IF_ERROR(GetHealthyHarness(request.servable_id, &harness));
    // Calling {Load,Unload}Request() synchronously here prevents any other
    // concurrent calls to Load/UnloadServable() from proceeding.
    switch (request.kind) {
      case LoadOrUnloadRequest::Kind::kLoad:
        TF_RETURN_IF_ERROR(harness->LoadRequested());
        break;
      case LoadOrUnloadRequest::Kind::kUnload:
        TF_RETURN_IF_ERROR(harness->UnloadRequested());
        break;
    }
    return Status::OK();
  }();
  if (!status.ok()) {
    done_callback(status);
    return;
  }
  load_unload_executor_->Schedule([this, request, done_callback]() {
    HandleLoadOrUnloadRequest(request, done_callback);
  });
}

void BasicManager::HandleLoadOrUnloadRequest(const LoadOrUnloadRequest& request,
                                             DoneCallback done_callback) {
  // Decision phase.
  Status decision_status;
  LoaderHarness* harness;
  {
    // We serialize the decision phases of the requests. We will make a decision
    // about the present request before allowing other requests to enter their
    // decision phase. See the .h file for more explanation and rationale.
    mutex_lock l(load_unload_decision_phase_mu_);
    decision_status = ApproveLoadOrUnload(request, &harness);
  }
  if (!decision_status.ok()) {
    done_callback(decision_status);
    return;
  }

  // Execution phase.
  const Status execution_status = ExecuteLoadOrUnload(request, harness);
  done_callback(execution_status);
}

Status BasicManager::ApproveLoadOrUnload(const LoadOrUnloadRequest& request,
                                         LoaderHarness** harness) {
  mutex_lock l(mu_);

  TF_RETURN_IF_ERROR(GetHealthyHarness(request.servable_id, harness));

  Status approval_status;
  switch (request.kind) {
    case LoadOrUnloadRequest::Kind::kLoad: {
      approval_status = ApproveLoad(*harness, &l);
      break;
    }
    case LoadOrUnloadRequest::Kind::kUnload: {
      approval_status = ApproveUnload(*harness);
      break;
    }
  }

  if (approval_status.ok()) {
    ++num_ongoing_load_unload_executions_;
  }

  return approval_status;
}

Status BasicManager::ApproveLoad(LoaderHarness* harness, mutex_lock* mu_lock) {
  if (resource_tracker_ != nullptr) {
    // Attempt to reserve resources for the load.
    while (true) {
      resource_tracker_->RecomputeUsedResources(
          GetLoadersCurrentlyUsingResources());
      bool resources_reserved;
      TF_RETURN_IF_ERROR(resource_tracker_->ReserveResources(
          *harness->loader(), &resources_reserved));
      if (resources_reserved) {
        // Woohoo! We got our resources.
        LOG(INFO) << "Successfully reserved resources to load servable "
                  << harness->id().DebugString();
        break;
      }

      // We weren't able to reserve the resources. See if there are any ongoing
      // load/unload executions that may be temporarily tying up resources.
      if (num_ongoing_load_unload_executions_ == 0) {
        // There are no ongoing load/unloads, so we really are out of resources
        // for this servable.
        LOG(WARNING) << "Unable to reserve resources to load servable "
                     << harness->id().DebugString();
        const Status error = errors::ResourceExhausted(
            "Insufficient resources to load servable ",
            harness->id().DebugString());
        harness->Error(error);
        PublishOnEventBus({harness->id(), ServableState::ManagerState::kEnd,
                           harness->status()});
        DeleteHarness(harness->id());
        return error;
      } else {
        // Wait until at least one load/unload request finishes, then retry.
        num_ongoing_load_unload_executions_cv_.wait(*mu_lock);
      }
    }
  }

  // Transition to state kLoadApproved inside the decision phase. We rely
  // on this state to know whether resources have been reserved in
  // GetLoadersCurrentlyUsingResources().
  TF_RETURN_IF_ERROR(harness->LoadApproved());

  return Status::OK();
}

Status BasicManager::ApproveUnload(LoaderHarness* harness) {
  // Transition to state kQuiescing inside the decision phase, to prevent any
  // concurrent unload requests from executing.
  //
  // StartQuiescing() returns an error status if the harness is not in a state
  // to be quiesced.
  TF_RETURN_IF_ERROR(harness->StartQuiescing());

  return Status::OK();
}

void BasicManager::PublishOnEventBus(const ServableState& state) {
  if (servable_event_bus_ != nullptr) {
    servable_event_bus_->Publish(state);
  }
}

}  // namespace serving
}  // namespace tensorflow
