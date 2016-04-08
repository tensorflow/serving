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

#ifndef TENSORFLOW_SERVING_CORE_DYNAMIC_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_DYNAMIC_MANAGER_H_

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
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/core/version_policy.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"
#include "tensorflow_serving/util/optional.h"
#include "tensorflow_serving/util/periodic_function.h"

namespace tensorflow {
namespace serving {

namespace internal {
class DynamicManagerTargetImpl;
}  // namespace internal

namespace test_util {
class DynamicManagerTestAccess;
}  // namespace test_util

// A DynamicManager is a Manager which can accept and manage
// changes to the list of aspired versions.
//
// The manager makes transitions between versions of a servable stream using a
// configured VersionPolicy. The manager prefers unloading before loading to
// free up resources in the server when deciding amongst transitions suggested
// by the policy.
//
// TODO(b/25631500): Make the DynamicManager configurable using a proto.
class DynamicManager final : public Manager,
                             public Target<std::unique_ptr<Loader>> {
 public:
  struct Options {
    // The periodicity, in microseconds, of the thread which manages the state
    // of the servables. Default: 100 milliseconds. If this is set less than or
    // equal to 0, we don't run this thread at all.
    int64 manage_state_interval_micros = 100 * 1000;

    Env* env = Env::Default();

    // The VersionPolicy to use for the manager. If unset, we check-fail.
    std::unique_ptr<VersionPolicy> version_policy;

    // EventBus to publish servable state changes. This is optional, if unset,
    // we don't publish.
    EventBus<ServableState>* servable_event_bus = nullptr;

    // Maximum number of times we try to load a servable before we give up.
    int max_num_load_tries = 5;

    // The interval, in microseconds, between each servable load retry.
    // Default: 1min.
    int64 load_retry_interval_micros = 1 * 60 * 1000 * 1000;
  };

  explicit DynamicManager(Options options);
  ~DynamicManager() override;

  std::vector<ServableId> ListAvailableServableIds() override;

  // Returns a callback to set the list of aspired versions for a particular
  // servable stream name, using Loaders. See the comments on
  // AspiredVersionsCallback in source.h.
  //
  // Below are some contract details pertaining specifically to DynamicManager.
  //
  // Load()/Unload() CALLS GO TO A SINGLE LOADER OBJECT:
  //
  // In general, multiple callback calls may supply a loader object for a given
  // servable id. Once the manager calls Load() on one of those loaders, its
  // next call for that id will be to the same loader's Unload() method. (In
  // other words, bracketed Load() and Unload() calls will be to the same loader
  // object.)
  //
  // NO SPONTANEOUS UNLOADING:
  //
  // The manager aims to evolve the loadedness states of the servable objects it
  // manages to match the aspired list, but at a given point in time the two may
  // not coincide. That is because (a) loading/unloading are not instantaneous
  // operations, (b) loading can fail, and (c) the manager reserves the right to
  // refuse to load a servable version in the aspired list e.g. due to resource
  // limitations.
  //
  // However, the manager does obey the following constraint: Once it has loaded
  // a given servable version V, as long as V is present in the latest aspired
  // list it cannot unload V. One purpose of this guarantee is to facilitate
  // incremental loading, in which version V's Load() implementation arranges to
  // copy state from (or share state with) and already-loaded version V-1 (or
  // any prior version(s) that are loaded, for that matter). As long as V-1 is
  // currently loaded, and remains part of the aspired list, V can rely on V-1
  // remaining loaded.
  Source<std::unique_ptr<Loader>>::AspiredVersionsCallback
  GetAspiredVersionsCallback() override;

 private:
  friend class internal::DynamicManagerTargetImpl;
  friend class test_util::DynamicManagerTestAccess;

  // Handles incoming aspired-versions requests from sources, via
  // 'target_impl_'.
  void SetAspiredVersions(
      const StringPiece servable_name,
      std::vector<ServableData<std::unique_ptr<Loader>>> versions)
      LOCKS_EXCLUDED(managed_map_mu_);

  // Unloads all harnesses. No destruction is done, that will be handled by the
  // shared_ptrs.
  void UnloadAllServables() LOCKS_EXCLUDED(managed_map_mu_);

  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  // Updates the serving map by copying servables from the managed map, which
  // are ready to be served.
  void UpdateServingMap() EXCLUSIVE_LOCKS_REQUIRED(managed_map_mu_);

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
      EXCLUSIVE_LOCKS_REQUIRED(managed_map_mu_);

  // Performs the action on the harness.
  void PerformAction(const VersionPolicy::Action action,
                     ManagedMap::iterator harness_iter)
      EXCLUSIVE_LOCKS_REQUIRED(managed_map_mu_);

  // Goes through the managed map and unloads the first servable it finds, which
  // is quiesced, and returns true. If no such servable is found, it returns
  // false without doing anything.
  bool UnloadQuiesced() EXCLUSIVE_LOCKS_REQUIRED(managed_map_mu_);

  // Goes through the harness map and calls the configured servable_policy with
  // the state snapshots to get a list of suggested actions. The actions are
  // then ordered and finally the topmost one is performed.
  optional<VersionPolicy::ServableAction> GetNextAction()
      EXCLUSIVE_LOCKS_REQUIRED(managed_map_mu_);

  // The  thread periodically runs this method.
  void ManageState() LOCKS_EXCLUDED(managed_map_mu_);

  // Publishes the state on the event bus, if an event bus was part of the
  // options, if not we ignore it.
  void PublishOnEventBus(const ServableState& state);

  const Options options_;

  mutable mutex managed_map_mu_;
  // ManagedMap contains all the servables managed by this manager, in different
  // states.
  ManagedMap managed_map_ GUARDED_BY(managed_map_mu_);

  // ServingMap contains all the servables which are ready to be served, which
  // is a subset of those in the managed map.
  // This map is updated occasionally from the main manager loop thread while
  // being accessed from multiple threads to get ServableHandles.
  class ServingMap {
   public:
    ServingMap();

    // Gets a list of all servable ids.
    std::vector<ServableId> ListAvailableServableIds();

    // Returns an UntypedServableHandle given a ServableRequest.
    // Returns error if no such Servable is available -- e.g. not yet loaded,
    // has been quiesced/unloaded, etc.
    Status GetUntypedServableHandle(
        const ServableRequest& request,
        std::unique_ptr<UntypedServableHandle>* const untyped_handle);

    // Returns a map of all the currently available servable_ids to their
    // corresponding UntypedServableHandles.
    std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
    GetAvailableUntypedServableHandles() const;

    // Updates the serving map by copying servables from the managed map, which
    // are ready to be served.
    void Update(const ManagedMap& managed_map,
                EventBus<ServableState>* servable_event_bus);

   private:
    struct EqRequest;
    // Hash and equality functors for ServableRequest.
    // Forward-declared here and defined in the cc file.
    struct HashRequest;

    // Map from ServableRequest to corresponding harness. For the
    // latest version of a servable stream, we add an extra entry for it, where
    // key is the ServableRequest without the version set, so that
    // requests for the latest, can be directly queried on this map.
    using HandlesMap =
        std::unordered_multimap<ServableRequest,
                                std::shared_ptr<const LoaderHarness>,
                                HashRequest, EqRequest>;
    FastReadDynamicPtr<HandlesMap> handles_map_;
  };
  ServingMap serving_map_;

  // Periodically runs the ManageState method in a background thread.
  std::unique_ptr<PeriodicFunction> manage_state_thread_;

  // The object that implements the Target API on behalf of this manager.
  std::unique_ptr<TargetBase<std::unique_ptr<Loader>>> target_impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicManager);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_DYNAMIC_MANAGER_H_
