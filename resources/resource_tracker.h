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

#ifndef TENSORFLOW_SERVING_RESOURCES_RESOURCE_TRACKER_H_
#define TENSORFLOW_SERVING_RESOURCES_RESOURCE_TRACKER_H_

#include <memory>
#include <vector>

#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/resources/resource_util.h"

namespace tensorflow {
namespace serving {

// A class that keeps track of the available and spoken-for resources in a
// serving system. It can decide whether enough resources are available to load
// a new servable.
//
// This class is not thread-safe.
class ResourceTracker {
 public:
  static Status Create(const ResourceAllocation& total_resources,
                       std::unique_ptr<ResourceUtil> util,
                       std::unique_ptr<ResourceTracker>* tracker);
  ~ResourceTracker() = default;

  // Determines whether enough resources are available to load 'servable', i.e.
  // is it guaranteed to fit in the gap between the used and total resources?
  // If so, adds the servable's resource allocation to the used resources and
  // sets 'success' to true. Otherwise, leaves the used resources unchanged and
  // sets 'success' to false. Upon encountering illegal data, e.g. if 'servable'
  // emits an invalid resource estimate, returns an error status.
  Status ReserveResources(const Loader& servable, bool* success);

  // Recomputes the used resources from scratch, given every loader whose
  // servable is either loaded or transitioning to/from being loaded,
  // specifically:
  //  * servables approved for loading (their resources are reserved),
  //  * servables in the process of loading,
  //  * servables that are currently loaded,
  //  * servables in the process of unloading.
  Status RecomputeUsedResources(const std::vector<const Loader*>& servables);

  const ResourceAllocation& total_resources() const { return total_resources_; }
  const ResourceAllocation& used_resources() const { return used_resources_; }

 private:
  ResourceTracker(const ResourceAllocation& total_resources,
                  std::unique_ptr<ResourceUtil> util);

  // A ResourceUtil object to use for operations and comparisons on allocations.
  const std::unique_ptr<ResourceUtil> util_;

  // The total resources the system has. Must be bound. Kept normalized.
  const ResourceAllocation total_resources_;

  // The resources currently set aside for servables that are loaded, or
  // transitioning to/from being loaded. May be bound or unbound. Kept
  // normalized.
  //
  // Under normal conditions, less than or equal to 'total_resources_'.
  ResourceAllocation used_resources_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceTracker);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_TRACKER_H_
