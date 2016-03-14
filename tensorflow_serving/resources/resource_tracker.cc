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

#include "tensorflow_serving/resources/resource_tracker.h"

#include <algorithm>
#include <string>

#include "tensorflow_serving/resources/resources.pb.h"

namespace tensorflow {
namespace serving {

ResourceTracker::ResourceTracker(const ResourceAllocation& total_resources,
                                 std::unique_ptr<ResourceUtil> util)
    : util_(std::move(util)), total_resources_(total_resources) {}

bool ResourceTracker::ReserveResources(const Loader& servable) {
  ResourceAllocation proposed_used_resources = used_resources_;
  const ResourceAllocation servable_resources = servable.EstimateResources();
  util_->Add(servable_resources, &proposed_used_resources);
  if (util_->LessThanOrEqual(proposed_used_resources, total_resources_)) {
    used_resources_ = proposed_used_resources;
    return true;
  } else {
    LOG(INFO) << "Insufficient resources to load servable "
              << "\ntotal resources:\n"
              << total_resources_.DebugString() << "used/reserved resources:\n"
              << used_resources_.DebugString()
              << "resources requested by servable:\n"
              << servable_resources.DebugString();
    return false;
  }
}

void ResourceTracker::RecomputeUsedResources(
    const std::vector<const Loader*>& servables) {
  used_resources_.Clear();
  for (const Loader* servable : servables) {
    util_->Add(servable->EstimateResources(), &used_resources_);
  }
}

}  // namespace serving
}  // namespace tensorflow
