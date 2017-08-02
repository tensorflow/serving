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
#include "tensorflow_serving/util/retrier.h"
#include "tensorflow_serving/util/threadpool_executor.h"

namespace tensorflow {
namespace serving {

namespace {

std::unique_ptr<Executor> CreateExecutor(Env* const env,
                                         const uint32 num_threads,
                                         const string& threadpool_name) {
  std::unique_ptr<Executor> executor;
  if (num_threads == 0) {
    executor.reset(new InlineExecutor());
  } else {
    executor.reset(new ThreadPoolExecutor(env, threadpool_name, num_threads));
  }
  return executor;
}

}  // namespace

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
  manager->reset(new BasicManager(
      options.env, options.num_load_threads, options.num_unload_threads,
      options.max_num_load_retries, options.load_retry_interval_micros,
      std::move(options.resource_tracker), options.servable_event_bus));
  return Status::OK();
}

BasicManager::BasicManager(Env* const env, const uint32 num_load_threads,
                           const uint32 num_unload_threads,
                           uint32 max_num_load_retries,
                           int64 load_retry_interval_micros,
                           std::unique_ptr<ResourceTracker> resource_tracker,
                           EventBus<ServableState>* servable_event_bus)
    : servable_event_bus_(servable_event_bus),
      env_(env),
      num_load_threads_(num_load_threads) {
  harness_options_.max_num_load_retries = max_num_load_retries;
  harness_options_.load_retry_interval_micros = load_retry_interval_micros;
  harness_options_.error_callback = [this](const ServableId& id,
                                           const Status& error) {
    PublishOnEventBus({id, ServableState::ManagerState::kEnd, error});
  };

  {
    mutex_lock l(num_load_threads_mu_);
    load_executor_ =
        CreateExecutor(env_, num_load_threads, "BasicManager_Load_ThreadPool");
  }
  unload_executor_ = CreateExecutor(env_, num_unload_threads,
                                    "BasicManager_Unload_ThreadPool");
  resource_tracker_ = std::move(resource_tracker);
}

BasicManager::~BasicManager() {
  // Reset the executors first to finish all pending loads/unloads.
  {
    mutex_lock l(num_load_threads_mu_);
    load_executor_.reset();
  }
  unload_executor_.reset();

  UnloadAllServables();
}

void BasicManager::UnloadAllServables() {
  LOG(INFO) << "Unload all remaining servables in the manager.";
  {
    mutex_lock l(mu_);
    for (auto it = managed_map_.begin(); it != managed_map_.end(); ++it) {
      LoaderHarness* const harness = it->second.get();
      if (harness->state() == LoaderHarness::State::kReady) {
        // TODO(b/35997855): Don't just ignore the ::tensorflow::Status object!
        harness->UnloadRequested().IgnoreError();
        harness->StartQuiescing().IgnoreError();
        harness->DoneQuiescing().IgnoreError();
        harness->Unload().IgnoreError();
      }
      if (harness->state() == LoaderHarness::State::kQuiescing) {
        // TODO(b/35997855): Don't just ignore the ::tensorflow::Status object!
        harness->DoneQuiescing().IgnoreError();
        harness->Unload().IgnoreError();
      }
      if (harness->state() == LoaderHarness::State::kQuiesced) {
        // TODO(b/35997855): Don't just ignore the ::tensorflow::Status object!
        harness->Unload().IgnoreError();
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

Status BasicManager::ManageServableInternal(
    ServableData<std::unique_ptr<Loader>> servable,
    std::function<std::shared_ptr<LoaderHarness>(const ServableId&,
                                                 std::unique_ptr<Loader>)>
        harness_creator) {
  VLOG(1) << "Request to start managing servable " << servable.id();

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
  } else {
    PublishOnEventBus({harness->id(), ServableState::ManagerState::kStart,
                       harness->status()});
  }
  managed_map_.emplace(servable.id().name, harness);

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

Status BasicManager::StopManagingServable(const ServableId& id) {
  VLOG(1) << "Request to stop managing servable " << id;
  mutex_lock l(mu_);
  const auto it = FindHarnessInMap(id);
  if (it == managed_map_.end()) {
    LOG(ERROR) << "Request to delete harness for " << id
               << ", but no such harness found in managed_map_";
    return errors::FailedPrecondition("This servable is not being managed: ",
                                      id.DebugString());
  }
  const auto state = it->second->state();
  if (state != LoaderHarness::State::kNew &&
      state != LoaderHarness::State::kError &&
      state != LoaderHarness::State::kDisabled) {
    LOG(ERROR) << "Request to delete harness for " << id
               << ", but it is not in a new or end state. State: " << state;
    return errors::FailedPrecondition(
        "This servable is not in a new or end state and we cannot stop "
        "managing it: ",
        id.DebugString(), " ", LoaderHarness::StateDebugString(state));
  }
  managed_map_.erase(it);
  return Status::OK();
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
  // We save the id of the harness so that we can publish it after Load(). (We
  // can't query harness again after Load() as it may be deleted by another
  // thread that called StopManagingServable().)
  const ServableId id = harness->id();

  // We don't hold the lock while calling Load() as it may block.
  TF_RETURN_IF_ERROR(harness->Load());

  {
    mutex_lock l(mu_);
    UpdateServingMap();
  }

  PublishOnEventBus(
      {id, ServableState::ManagerState::kAvailable, Status::OK()});
  return Status::OK();
}

void BasicManager::LoadServable(const ServableId& id,
                                const DoneCallback done_callback) {
  VLOG(1) << "Request to load servable " << id;
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
  // We save the id of the harness so that we can publish it after Unload(). (We
  // can't query harness again after Unload() as it may be deleted by another
  // thread that called StopManagingServable().)
  const ServableId id = harness->id();

  {
    // StartQuiescing() would have been already called.
    mutex_lock l(mu_);
    PublishOnEventBus(
        {id, ServableState::ManagerState::kUnloading, harness->status()});
    UpdateServingMap();
    TF_RETURN_IF_ERROR(harness->DoneQuiescing());
  }

  // We don't hold the lock while calling Unload() as it may block.
  TF_RETURN_IF_ERROR(harness->Unload());
  PublishOnEventBus({id, ServableState::ManagerState::kEnd, Status::OK()});
  return Status::OK();
}

void BasicManager::UnloadServable(const ServableId& id,
                                  const DoneCallback done_callback) {
  VLOG(1) << "Request to unload servable " << id;
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

void BasicManager::SetNumLoadThreads(const uint32 num_load_threads) {
  mutex_lock l(num_load_threads_mu_);

  load_executor_.reset();
  num_load_threads_ = num_load_threads;
  load_executor_ =
      CreateExecutor(env_, num_load_threads_, "BasicManager_Load_ThreadPool");
}

uint32 BasicManager::num_load_threads() const {
  mutex_lock l(num_load_threads_mu_);

  return num_load_threads_;
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

  switch (request.kind) {
    case LoadOrUnloadRequest::Kind::kLoad: {
      mutex_lock l(num_load_threads_mu_);
      load_executor_->Schedule([this, request, done_callback]() {
        HandleLoadOrUnloadRequest(request, done_callback);
      });
      break;
    }
    case LoadOrUnloadRequest::Kind::kUnload: {
      unload_executor_->Schedule([this, request, done_callback]() {
        HandleLoadOrUnloadRequest(request, done_callback);
      });
      break;
    }
  }
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

  switch (request.kind) {
    case LoadOrUnloadRequest::Kind::kLoad: {
      TF_RETURN_IF_ERROR(ApproveLoad(*harness, &l));
      break;
    }
    case LoadOrUnloadRequest::Kind::kUnload: {
      TF_RETURN_IF_ERROR(ApproveUnload(*harness));
      break;
    }
  }

  ++num_ongoing_load_unload_executions_;

  return Status::OK();
}

Status BasicManager::ApproveLoad(LoaderHarness* harness, mutex_lock* mu_lock) {
  if (resource_tracker_ != nullptr) {
    // Attempt to reserve resources for the load.
    const Status resource_reservation_status =
        ReserveResources(harness, mu_lock);
    if (!resource_reservation_status.ok()) {
      LOG(WARNING) << resource_reservation_status;
      harness->Error(resource_reservation_status);
      PublishOnEventBus({harness->id(), ServableState::ManagerState::kEnd,
                         resource_reservation_status});
      return resource_reservation_status;
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
  TF_RETURN_IF_ERROR(harness->StartQuiescing());

  return Status::OK();
}

Status BasicManager::ReserveResources(LoaderHarness* harness,
                                      mutex_lock* mu_lock) {
  while (true) {
    // TODO(b/35997855): Don't just ignore the ::tensorflow::Status object!
    resource_tracker_
        ->RecomputeUsedResources(GetLoadersCurrentlyUsingResources())
        .IgnoreError();
    bool resources_reserved;
    // We retry reserving resources because it may involve transiently failing
    // operations like file-reads.
    const Status reserve_resources_status =
        Retry(strings::StrCat("Reserving resources for servable: ",
                              harness->id().DebugString()),
              harness_options_.max_num_load_retries,
              harness_options_.load_retry_interval_micros,
              [&]() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                return resource_tracker_->ReserveResources(*harness->loader(),
                                                           &resources_reserved);
              },
              [&]() { return harness->cancel_load_retry(); });
    if (!reserve_resources_status.ok()) {
      return errors::Internal(strings::StrCat(
          "Error while attempting to reserve resources to load servable ",
          harness->id().DebugString(), ": ",
          reserve_resources_status.error_message()));
    }
    if (resources_reserved) {
      // Woohoo! We got our resources.
      LOG(INFO) << "Successfully reserved resources to load servable "
                << harness->id().DebugString();
      return Status::OK();
    }

    // We weren't able to reserve the resources. See if there are any
    // ongoing load/unload executions that may be temporarily tying up
    // resources.
    if (num_ongoing_load_unload_executions_ == 0) {
      // There are no ongoing load/unloads, so we really are out of
      // resources for this servable.
      return errors::ResourceExhausted(
          "Insufficient resources to load servable ",
          harness->id().DebugString());
    } else {
      // Wait until at least one load/unload request finishes, then retry.
      VLOG(1) << "Waiting for another load/unload request to finish";
      num_ongoing_load_unload_executions_cv_.wait(*mu_lock);
    }
  }
}

void BasicManager::PublishOnEventBus(const ServableState& state) {
  if (servable_event_bus_ != nullptr) {
    servable_event_bus_->Publish(state);
  }
}

}  // namespace serving
}  // namespace tensorflow
