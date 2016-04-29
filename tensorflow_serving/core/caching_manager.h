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

namespace tensorflow {
namespace serving {

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
    // request.
    virtual Status CreateLoader(
        const ServableRequest& request,
        std::unique_ptr<ServableData<std::unique_ptr<Loader>>>*
            loader_data) = 0;
  };

  static Status Create(Options options,
                       std::unique_ptr<LoaderFactory> loader_factory,
                       std::unique_ptr<CachingManager>* caching_manager);

  ~CachingManager() override;

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override;

  std::vector<ServableId> ListAvailableServableIds() const override;

 private:
  CachingManager(std::unique_ptr<LoaderFactory> loader_factory,
                 std::unique_ptr<BasicManager> basic_manager);

  // Returns the untyped handle for the servable request.
  //
  // Semantics related to a ServableRequest for "latest":
  // 1. If the manager does not have any version of the requested servable
  // loaded, it forwards the "latest" request to the loader-factory, which emits
  // its notion of the "latest" version. This is then managed and loaded by the
  // manager.
  // 2. If the manager already has one or more versions of the requested
  // servable name, it will return the latest among those versions (regardless
  // of whether the loader-factory may know of a later version).
  //
  // TODO(b/25449742): Always return latest as determined by the loader factory.
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* handle) override;

  std::unique_ptr<LoaderFactory> loader_factory_;

  std::unique_ptr<BasicManager> basic_manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(CachingManager);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_CACHING_MANAGER_H_
