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

#ifndef TENSORFLOW_SERVING_CORE_BASIC_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_BASIC_MANAGER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/resources/resource_tracker.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/executor.h"
#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// Helps manage the lifecycle of servables including loading, serving and
// unloading them. The manager accepts servables in the form of Loaders.
//
// We start managing a servable through one of the ManageServable* methods. You
// can go on to load the servable after this by calling LoadServable. Loading
// will also make the servable available to serve. Once you decide to unload it,
// you can call UnloadServable on it, which will make it unavailable to serve,
// then unload the servable and delete it. After unload, the servable is no
// longer managed by the manager.
//
// BasicManager tracks the resources (e.g. RAM) used by loaded servables, and
// only allows loading new servables that fit within the overall resource pool.
//
// BasicManager can be configured to use a thread-pool to do it's load and
// unloads. This makes the {Load,Unload}Servable() methods schedule the
// load/unloads rather than executing them synchronously. If there are more
// pending load/unloads than threads in the thread pool, they are processed in
// FIFO order.
//
// In the presence of loaders that over-estimate their servables' resource needs
// and/or only bind their servables' resources to device instances, load/unload
// concurrency can be reduced below the thread-pool size. That is because we may
// have to wait for one servable's load/unload to finish to pin down the
// resource availability for loading another servable.
//
// REQUIRES:
// 1. Order of method calls -
//    ManageServable*() -> LoadServable() -> UnloadServable().
// 2. Do not schedule concurrent load and unloads of the same servable.
// 3. Do not call load or unload multiple times on the same servable.
//
// This class is thread-safe.
//
// Example usage:
//
// const ServableId id = {kServableName, 0};
// std::unique_ptr<Loader> loader = ...;
// ...
// BasicManager manager;
// TF_CHECK_OK(manager.ManageServable(
//     CreateServableData(id, std::move(loader))));
// TF_CHECK_OK(manager.LoadServable(id));
//
// ...
// TF_CHECK_OK(manager.GetServableHandle(
//     ServableRequest::Latest(kServableName), &handle));
// ...
//
// TF_CHECK_OK(manager.UnloadServable(id));
class BasicManager : public Manager {
 public:
  struct Options {
    // The resource tracker to use while managing servable resources. Optional.
    // If left as nullptr, we do not validate servable resource usage.
    std::unique_ptr<ResourceTracker> resource_tracker;

    // The number of threads in the thread-pool used to load and unload
    // servables.
    //
    // If set as 0, we don't use a thread-pool, and the {Load,Unload}Servable()
    // methods block.
    uint32 num_load_unload_threads = 0;

    // EventBus to publish servable state changes. This is optional, if unset,
    // we don't publish.
    EventBus<ServableState>* servable_event_bus = nullptr;

    // Maximum number of times we retry loading a servable, after the first
    // failure, before we give up.
    //
    // If set to 0, a load is attempted only once.
    uint32 max_num_load_retries = 5;

    // The interval, in microseconds, between each servable load retry. If set
    // negative, we don't wait.
    // Default: 1 minute.
    int64 load_retry_interval_micros = 1LL * 60 * 1000 * 1000;

    // The environment to use for starting threads in the thread-pool.
    Env* env = Env::Default();
  };
  static Status Create(Options options, std::unique_ptr<BasicManager>* manager);

  // If configured to use a load/unload thread-pool, waits until all scheduled
  // loads and unloads have finished and then destroys the set of threads.
  ~BasicManager() override;

  std::vector<ServableId> ListAvailableServableIds() const override;

  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  // Starts managing the servable.
  //
  // If called multiple times with the same servable id, all of them are
  // accepted, but only the first one is used. We accept the servable even if
  // called with erroneous ServableData.
  void ManageServable(ServableData<std::unique_ptr<Loader>> servable);

  // Similar to the above method, but callers, usually other managers built on
  // top of this one, can associate additional state with the servable.
  // Additional state may be ACL or lifetime metadata for that servable.  The
  // ownership of the state is transferred to this class.
  template <typename T>
  void ManageServableWithAdditionalState(
      ServableData<std::unique_ptr<Loader>> servable,
      std::unique_ptr<T> additional_state);

  // Returns the names of all the servables managed by this manager. The names
  // will be duplicate-free and not in any particular order.
  std::vector<string> GetManagedServableNames() const;

  // Returns the state snapshots of all the servables of a particular stream,
  // managed by this manager.
  //
  // T is the additional-state type, if any.
  template <typename T = std::nullptr_t>
  std::vector<ServableStateSnapshot<T>> GetManagedServableStateSnapshots(
      const string& servable_name) const;

  // Returns the state snapshot of a particular servable-id managed by this
  // manager if available.
  //
  // REQUIRES: This manager should have been managing this servable already,
  // else we return nullopt.
  template <typename T = std::nullptr_t>
  optional<ServableStateSnapshot<T>> GetManagedServableStateSnapshot(
      const ServableId& id);

  // Returns the additional state for the servable. Returns nullptr if there is
  // no additional state setup or if there is a type mismatch between what was
  // setup and what is being asked for.
  //
  // REQUIRES: This manager should have been managing this servable already,
  // else we return nullptr.
  template <typename T>
  T* GetAdditionalServableState(const ServableId& id);

  // Callback called at the end of {Load,Unload}Servable(). We pass in the
  // status of the operation to the callback.
  using DoneCallback = std::function<void(const Status& status)>;

  // Loads the servable with this id, and updates the serving map too. Calls
  // 'done_callback' with ok iff the servable was loaded successfully, else
  // returns an error status.
  //
  // If using a thread-pool, this method transitions the servable harness to
  // kLoading state, schedules the load and returns, otherwise it
  // completes the load before returning.
  //
  // REQUIRES: This manager should have been managing this servable already, for
  // it to be loaded, else we call 'done_callback' with an error status. Do not
  // call this multiple times on the same servable. Only one of those will
  // succeed and the rest will fail with an error status.
  void LoadServable(const ServableId& id, DoneCallback done_callback);

  // Cancels retrying the servable load during LoadServable(). Does nothing if
  // the servable isn't managed.
  //
  // If the retries are cancelled, the servable goes into a state dependent on
  // the last Load() called on it. If the last Load() was successful, it will be
  // in state kReady, else in kError.
  void CancelLoadServableRetry(const ServableId& id);

  // Unloads the servable with this id, and updates the serving map too. Calls
  // 'done_callback' with ok iff the servable was unloaded successfully, else
  // returns an error status.
  //
  // If using a thread-pool, this method transitions the servable harness to
  // kQuiescing state, schedules the unload and returns, otherwise it completes
  // the unload before returning.
  //
  // REQUIRES: This manager should have been managing this servable already, for
  // it to be unloaded, else calls 'done_callback' with an error status. Do not
  // call this multiple times on the same servable. Only one of those will
  // succeed and the rest will fail with an error status.
  void UnloadServable(const ServableId& id, DoneCallback done_callback);

 private:
  BasicManager(std::unique_ptr<Executor> load_unload_executor,
               std::unique_ptr<ResourceTracker> resource_tracker,
               EventBus<ServableState>* servable_event_bus,
               const LoaderHarness::Options& harness_options);

  // Starts managing the servable.
  //
  // If called multiple times with the same servable id, all of them are
  // accepted, but only the first one is used. We accept the servable even if
  // called with erroneous ServableData.
  //
  // Also accepts a closure to create the harness as a shared_ptr. The harness
  // has a different constructors for creating it with or without
  // additional_state.
  void ManageServableInternal(ServableData<std::unique_ptr<Loader>> servable,
                              std::function<std::shared_ptr<LoaderHarness>(
                                  const ServableId&, std::unique_ptr<Loader>)>
                                  harness_creator);

  // Obtains the harness associated with the given servable id. Returns an ok
  // status if a corresponding harness was found, else an error status.
  Status GetHealthyHarness(const ServableId& servable_id,
                           LoaderHarness** harness)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Obtains a pointer to every managed loader that is currently holding
  // resources, i.e. whose state is one of kApprovedForLoading, kLoading,
  // kReady, kUnloadRequested, kQuiescing, kQuiesced or kUnloading.
  std::vector<const Loader*> GetLoadersCurrentlyUsingResources() const
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // A load or unload request for a particular servable. Facilitates code
  // sharing across the two cases.
  struct LoadOrUnloadRequest {
    enum class Kind { kLoad, kUnload };
    Kind kind;
    ServableId servable_id;
  };

  // A unification of LoadServable() and UnloadServable().
  void LoadOrUnloadServable(const LoadOrUnloadRequest& request,
                            DoneCallback done_callback) LOCKS_EXCLUDED(mu_);

  // The synchronous logic for handling a load/unload request, including both
  // the decision and execution phases. This is the method run in the executor.
  void HandleLoadOrUnloadRequest(const LoadOrUnloadRequest& request,
                                 DoneCallback done_callback)
      LOCKS_EXCLUDED(mu_);

  // The decision phase of whether to approve a load/unload request. Delegates
  // to one of ApproveLoad() or ApproveUnload() -- see those methods' comments
  // for details.
  //
  // Upon approving the request, signals entrance to the execution phase by
  // incrementing 'num_ongoing_load_unload_executions_'.
  //
  // If returning "ok", populates 'harness' with the harness for the request's
  // servable. (Note that 'harness' is guaranteed to remain live for the
  // subsequent execution phase of the request because approval of this request
  // precludes concurrent execution of another request that could delete the
  // harness.)
  Status ApproveLoadOrUnload(const LoadOrUnloadRequest& request,
                             LoaderHarness** harness) LOCKS_EXCLUDED(mu_);

  // The decision phase of whether to approve a load request. If it succeeds,
  // places the servable into state kApprovedForLoad. Among other things, that
  // prevents a subsequent load request from proceeding concurrently.
  //
  // Argument 'mu_lock' is a lock held on 'mu_'. It is released temporarily via
  // 'num_ongoing_load_unload_executions_cv_'.
  Status ApproveLoad(LoaderHarness* harness, mutex_lock* mu_lock)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The decision phase of whether to approve an unload request. If it succeeds,
  // places the servable into state kQuiescing. Among other things, that
  // prevents a subsequent unload request from proceeding concurrently.
  Status ApproveUnload(LoaderHarness* harness) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The execution phase of loading/unloading a servable. Delegates to either
  // ExecuteLoad() or ExecuteUnload().
  //
  // Upon completion (and regardless of the outcome), signals exit of the
  // execution phase by decrementing 'num_ongoing_load_unload_executions_'.
  Status ExecuteLoadOrUnload(const LoadOrUnloadRequest& request,
                             LoaderHarness* harness);

  // The execution phase of loading a servable.
  Status ExecuteLoad(LoaderHarness* harness) LOCKS_EXCLUDED(mu_);

  // The execution phase of loading a unservable.
  Status ExecuteUnload(LoaderHarness* harness) LOCKS_EXCLUDED(mu_);

  // Unloads all the managed servables.
  void UnloadAllServables() LOCKS_EXCLUDED(mu_);

  // Updates the serving map by copying servables from the managed map, which
  // are ready to be served.
  void UpdateServingMap() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  struct HashString {
    uint64 operator()(const string& str) const { return Hash64(str); }
  };
  // Keys are the servable names.
  // Values are the harnesses for each servable version. The values when
  // fetched, are unordered.
  using ManagedMap =
      std::unordered_multimap<string, std::shared_ptr<LoaderHarness>,
                              HashString>;

  // Fetches the harness with this id from the harness_map_. Returns
  // harness_map_.end(), if the harness is not found.
  ManagedMap::iterator FindHarnessInMap(const ServableId& id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Publishes the state on the event bus, if an event bus was part of the
  // options, if not we ignore it.
  void PublishOnEventBus(const ServableState& state);

  const LoaderHarness::Options harness_options_;

  // The event bus to which to publish servable state change events, or nullptr
  // if no bus has been configured.
  EventBus<ServableState>* servable_event_bus_;

  // Used to protect access to 'managed_map_', 'resource_tracker_' and other
  // core state elements.
  mutable mutex mu_;

  // ManagedMap contains all the servables managed by this manager, in different
  // states.
  ManagedMap managed_map_ GUARDED_BY(mu_);

  // ServingMap contains all the servables which are ready to be served, which
  // is a subset of those in the managed map.
  // This map is updated occasionally from the main manager loop thread while
  // being accessed from multiple threads to get ServableHandles.
  //
  // This class is thread-safe.
  class ServingMap {
   public:
    ServingMap();

    // Gets a list of all servable ids.
    std::vector<ServableId> ListAvailableServableIds() const;

    // Returns an UntypedServableHandle given a ServableRequest.
    // Returns error if no such Servable is available -- e.g. not yet loaded,
    // has been quiesced/unloaded, etc.
    Status GetUntypedServableHandle(
        const ServableRequest& request,
        std::unique_ptr<UntypedServableHandle>* untyped_handle);

    // Returns a map of all the currently available servable_ids to their
    // corresponding UntypedServableHandles.
    std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
    GetAvailableUntypedServableHandles() const;

    // Updates the serving map by copying servables from the managed map, which
    // are ready to be served.
    void Update(const ManagedMap& managed_map);

   private:
    struct EqRequest;
    // Hash and equality functors for ServableRequest.
    // Forward-declared here and defined in the cc file.
    struct HashRequest;

    // Map from ServableRequest to corresponding harness. For the latest version
    // of a servable stream, we add an extra entry for it, where key is the
    // ServableRequest without the version set, so that requests for the latest,
    // can be directly queried on this map.
    using HandlesMap =
        std::unordered_multimap<ServableRequest,
                                std::shared_ptr<const LoaderHarness>,
                                HashRequest, EqRequest>;
    FastReadDynamicPtr<HandlesMap> handles_map_;
  };
  ServingMap serving_map_;

  ////////
  // State associated with loading/unloading servables, and tracking their
  // resources.
  //
  // Load/unload requests have two phases: a decision phase and an execution
  // phase. The decision phase either accepts or rejects the request; if
  // accepted the execution phase executes the request (i.e. invokes Load() or
  // Unload() on the servable's loader).
  //
  // Given a stream of load/unload requests, we execute the decision phases
  // serially, which guarantees that request i’s decision phase can complete
  // before considering request i+1's so there’s no starvation.

  // The executor used for executing load and unload of servables.
  std::unique_ptr<Executor> load_unload_executor_;

  // Used to serialize the decision phases of the load/unload requests.
  mutable mutex load_unload_decision_phase_mu_;

  // A module that keeps track of available, used and reserved servable
  // resources (e.g. RAM).
  std::unique_ptr<ResourceTracker> resource_tracker_ GUARDED_BY(mu_);

  // The number of load/unload requests currently in their execution phase.
  int num_ongoing_load_unload_executions_ GUARDED_BY(mu_) = 0;

  // Used to wake up threads that are waiting for 'num_ongoing_executions' to
  // decrease.
  condition_variable num_ongoing_load_unload_executions_cv_;

  TF_DISALLOW_COPY_AND_ASSIGN(BasicManager);
};

////
// Implementation details. API readers may skip.
////

template <typename T>
void BasicManager::ManageServableWithAdditionalState(
    ServableData<std::unique_ptr<Loader>> servable,
    std::unique_ptr<T> additional_state) {
  ManageServableInternal(
      std::move(servable),
      [this, &additional_state](const ServableId& id,
                                std::unique_ptr<Loader> loader) {
        return std::make_shared<LoaderHarness>(id, std::move(loader),
                                               std::move(additional_state),
                                               harness_options_);
      });
}

template <typename T>
std::vector<ServableStateSnapshot<T>>
BasicManager::GetManagedServableStateSnapshots(
    const string& servable_name) const {
  mutex_lock l(mu_);

  const auto range = managed_map_.equal_range(servable_name);
  std::vector<ServableStateSnapshot<T>> state_snapshots;
  state_snapshots.reserve(std::distance(range.first, range.second));
  for (auto it = range.first; it != range.second; ++it) {
    state_snapshots.push_back(it->second->loader_state_snapshot<T>());
  }

  return state_snapshots;
}

template <typename T>
optional<ServableStateSnapshot<T>>
BasicManager::GetManagedServableStateSnapshot(const ServableId& id) {
  mutex_lock l(mu_);

  auto iter = FindHarnessInMap(id);
  if (iter == managed_map_.end()) {
    return nullopt;
  }
  return iter->second->loader_state_snapshot<T>();
}

template <typename T>
T* BasicManager::GetAdditionalServableState(const ServableId& id) {
  mutex_lock l(mu_);

  auto iter = FindHarnessInMap(id);
  if (iter == managed_map_.end()) {
    DCHECK(false) << "This servable is not being managed by the mananger: "
                  << id.DebugString();
    return nullptr;
  }
  return iter->second->additional_state<T>();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_BASIC_MANAGER_H_
