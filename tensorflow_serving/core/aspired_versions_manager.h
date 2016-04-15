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

#ifndef TENSORFLOW_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_H_

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
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/basic_manager.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/util/optional.h"
#include "tensorflow_serving/util/periodic_function.h"

namespace tensorflow {
namespace serving {

namespace internal {
class AspiredVersionsManagerTargetImpl;
}  // namespace internal

namespace test_util {
class AspiredVersionsManagerTestAccess;
}  // namespace test_util

// A manager that implements the Target<Loader> API which uses aspired-versions
// callbacks to dictate which servables to load/unload.
//
// The manager makes transitions between versions of a servable stream using a
// configured AspiredVersionPolicy. The manager prefers unloading before loading
// to free up resources in the server when deciding amongst transitions
// suggested by the policy.
class AspiredVersionsManager : public Manager,
                               public Target<std::unique_ptr<Loader>> {
 public:
  struct Options {
    // The resource tracker to use while managing servable resources. Optional.
    // If left as nullptr, we do not validate servable resource usage.
    std::unique_ptr<ResourceTracker> resource_tracker;

    // The periodicity, in microseconds, of the thread which manages the state
    // of the servables. Default: 100 milliseconds. If this is set less than or
    // equal to 0, we don't run this thread at all.
    int64 manage_state_interval_micros = 100 * 1000;

    // EventBus to publish servable state changes. This is optional, if unset,
    // we don't publish.
    EventBus<ServableState>* servable_event_bus = nullptr;

    // The AspiredVersionPolicy to use for the manager. Must be non-null.
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy;

    // The number of threads in the thread-pool used to load and unload
    // servables.
    //
    // If set as 0, we don't use a thread-pool, and the {Load,Unload}Servable()
    // methods block.
    uint32 num_load_unload_threads = 0;

    // Maximum number of times we retry loading a servable, after the first
    // failure, before we give up.
    uint32 max_num_load_retries = 5;

    // The interval, in microseconds, between each servable load retry. If set
    // negative, we don't wait.
    // Default: 1 minute.
    int64 load_retry_interval_micros = 1LL * 60 * 1000 * 1000;

    // The environment to use for starting threads in the thread-pool or for
    // sleeping.
    Env* env = Env::Default();
  };
  static Status Create(Options options,
                       std::unique_ptr<AspiredVersionsManager>* manager);
  ~AspiredVersionsManager() override;

  std::vector<ServableId> ListAvailableServableIds() override;

  // Returns a callback to set the list of aspired versions for a particular
  // servable stream name, using Loaders. See the comments on
  // AspiredVersionsCallback in source.h.
  //
  // Below are some contract details pertaining specifically to
  // AspiredVersionsManager.
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
  friend class internal::AspiredVersionsManagerTargetImpl;
  friend class test_util::AspiredVersionsManagerTestAccess;

  AspiredVersionsManager(
      int64 manage_state_interval_micros, Env* env,
      std::unique_ptr<AspiredVersionPolicy> aspired_version_policy,
      std::unique_ptr<BasicManager> basic_manager);

  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  // Handles incoming aspired-versions requests from sources, via
  // 'target_impl_'.
  void SetAspiredVersions(
      const StringPiece servable_name,
      std::vector<ServableData<std::unique_ptr<Loader>>> versions)
      LOCKS_EXCLUDED(basic_manager_read_modify_write_mu_);

  // Performs the action on the harness.
  void PerformAction(const AspiredVersionPolicy::ServableAction action)
      EXCLUSIVE_LOCKS_REQUIRED(basic_manager_read_modify_write_mu_);

  // Goes through the harness map and calls the configured servable_policy with
  // the state snapshots to get a list of suggested actions. The actions are
  // then ordered and finally the topmost one is performed.
  optional<AspiredVersionPolicy::ServableAction> GetNextAction()
      EXCLUSIVE_LOCKS_REQUIRED(basic_manager_read_modify_write_mu_);

  // Manages the state of the servables periodically.
  void ManageState() LOCKS_EXCLUDED(basic_manager_read_modify_write_mu_);

  std::unique_ptr<AspiredVersionPolicy> aspired_version_policy_;

  // To lock basic_manager_ to perform atomic read/modify/write operations on
  // the set of managed servables and their state (in particular, aspiredness).
  mutable mutex basic_manager_read_modify_write_mu_;

  // Periodically runs the ManageState method in a background thread.
  std::unique_ptr<PeriodicFunction> manage_state_thread_;

  // The object that implements the Target API on behalf of this manager.
  std::unique_ptr<TargetBase<std::unique_ptr<Loader>>> target_impl_;

  // This is where the servables "live" while they are being managed.
  std::unique_ptr<BasicManager> basic_manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(AspiredVersionsManager);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_ASPIRED_VERSIONS_MANAGER_H_
