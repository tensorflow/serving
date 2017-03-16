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

#ifndef TENSORFLOW_SERVING_CORE_CACHING_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_CACHING_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow_serving/core/basic_manager.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/source_adapter.h"

namespace tensorflow {
namespace serving {

namespace test_util {
class CachingManagerTestAccess;
}  // namespace test_util

// A manager that manages and loads servables on-demand. Upon receiving the
// request for a servable name and optional version, the manager checks if it
// already has the requested servable loaded. If not, it initiates the load
// operation and then serves the request.
//
// The manager blocks on the load operation and returns the handle when the
// servable has been loaded, or upon error.
//
// TODO(b/25449742): Add support for evictions of loaded servables from the
// caching-manager.
class CachingManager : public Manager {
 public:
  struct Options {
    // The resource tracker to use while managing servable resources. Optional.
    // If left as nullptr, we do not validate servable resource usage.
    std::unique_ptr<ResourceTracker> resource_tracker;

    // The number of threads in the thread-pool used to load servables.
    //
    // If set as 0, we don't use a thread-pool, and LoadServable() blocks.
    uint32 num_load_threads = 0;

    // The number of threads in the thread-pool used to unload servables.
    //
    // If set as 0, we don't use a thread-pool.
    uint32 num_unload_threads = 0;

    // EventBus to publish servable state changes. This is optional, if unset,
    // we don't publish.
    EventBus<ServableState>* servable_event_bus = nullptr;

    // Maximum number of times we retry loading a servable, after the first
    // failure, before we give up. If set to 0, a load is attempted only once.
    uint32 max_num_load_retries = 5;

    // The interval, in microseconds, between each servable load retry. If set
    // negative, we don't wait.
    // Default: 1 minute.
    int64 load_retry_interval_micros = 1LL * 60 * 1000 * 1000;

    // The environment to use for starting threads in the thread-pool.
    Env* env = Env::Default();
  };

  // An abstraction for a loader-factory to map from a servable request to the
  // corresponding loader.
  class LoaderFactory {
   public:
    virtual ~LoaderFactory() = default;

    // Creates servable data consisting of the loader corresponding to the
    // servable-id. Any errors can be reported by embedding them in the returned
    // ServableData item.
    virtual ServableData<std::unique_ptr<Loader>> CreateLoader(
        const ServableId& servable_id) = 0;

    // Returns the latest version corresponding to the servable name.
    virtual int64 GetLatestVersion(const string& servable_name) const = 0;
  };

  static Status Create(Options options,
                       std::unique_ptr<LoaderFactory> loader_factory,
                       std::unique_ptr<CachingManager>* caching_manager);

  ~CachingManager() override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  std::vector<ServableId> ListAvailableServableIds() const override;

 private:
  friend class test_util::CachingManagerTestAccess;

  CachingManager(std::unique_ptr<LoaderFactory> loader_factory,
                 std::unique_ptr<BasicManager> basic_manager);

  // Returns the untyped handle for the servable request.
  //
  // Semantics related to a ServableRequest for "latest":
  // The manager forwards the "latest" request to the loader-factory, which
  // emits its notion of the "latest" version. This is then managed and loaded
  // by the manager, if not already available, and a handle to it is returned.
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* handle) override;

  // Returns the untyped handle for a servable-id.
  Status GetUntypedServableHandleForId(
      const ServableId& servable_id,
      std::unique_ptr<UntypedServableHandle>* handle);

  // Transfer the given servable to 'basic_manager_', and ask it to load it. For
  // multiple concurrent requests for the same servable-id, enforces that
  // exactly one thread performs the load operation using the wrapped
  // basic-manager. All other requests block until the load completes and then
  // trivially succeed.
  Status LoadServable(ServableData<std::unique_ptr<Loader>> loader_data)
      LOCKS_EXCLUDED(load_mutex_map_mu_);

  // Returns the size of the load_mutex_map_.
  int64 GetLoadMutexMapSize() const LOCKS_EXCLUDED(load_mutex_map_mu_);

  // Erases the entry from the map corresponding to the servable-id if there is
  // only one remaining reference to the mutex.
  void MaybeEraseLoadMutexMapEntry(const ServableId& servable_id);

  std::unique_ptr<LoaderFactory> loader_factory_;

  std::unique_ptr<BasicManager> basic_manager_;

  // Used to protect access to the load_mutex_map_.
  mutable mutex load_mutex_map_mu_;

  // Map of servable-id to a mutex, which is required to synchronize calls to
  // load the servable using the wrapped basic-manager. The value in the map is
  // a shared_ptr to allow for reference counting and consequent garbage
  // collection.
  std::map<ServableId, std::shared_ptr<mutex>> load_mutex_map_
      GUARDED_BY(load_mutex_map_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(CachingManager);
};

// A simple LoaderFactory that looks for a servable at a path formed by
// concatenating a fixed path prefix with the servable's name. It assumes that
// a given servable only has one version, namely version 0.
class PathPrefixLoaderFactory : public CachingManager::LoaderFactory {
 public:
  PathPrefixLoaderFactory(const string& path_prefix,
                          std::unique_ptr<StoragePathSourceAdapter> adapter);
  ~PathPrefixLoaderFactory() override = default;

  ServableData<std::unique_ptr<Loader>> CreateLoader(
      const ServableId& id) override;

  int64 GetLatestVersion(const string& servable_name) const override;

 private:
  // The prefix of the path to the servables.
  const string path_prefix_;

  // An adapter for creating a loader from a given path.
  const std::unique_ptr<StoragePathSourceAdapter> adapter_;

  TF_DISALLOW_COPY_AND_ASSIGN(PathPrefixLoaderFactory);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_CACHING_MANAGER_H_
