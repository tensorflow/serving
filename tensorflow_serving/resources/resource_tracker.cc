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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/resources/resources.pb.h"

namespace tensorflow {
namespace serving {

Status ResourceTracker::Create(const ResourceAllocation& total_resources,
                               std::unique_ptr<ResourceUtil> util,
                               std::unique_ptr<ResourceTracker>* tracker) {
  TF_RETURN_IF_ERROR(util->VerifyValidity(total_resources));
  const ResourceAllocation normalized_total_resources =
      util->Normalize(total_resources);
  if (!util->IsBound(normalized_total_resources)) {
    return errors::InvalidArgument("total_resources must be bound: ",
                                   total_resources.DebugString());
  }
  tracker->reset(
      new ResourceTracker(normalized_total_resources, std::move(util)));
  return Status::OK();
}

Status ResourceTracker::ReserveResources(const Loader& servable,
                                         bool* success) {
  ResourceAllocation servable_resources;
  TF_RETURN_IF_ERROR(servable.EstimateResources(&servable_resources));
  TF_RETURN_IF_ERROR(util_->VerifyValidity(servable_resources));
  servable_resources = util_->Normalize(servable_resources);

  ResourceAllocation conservative_proposed_used_resources =
      util_->Overbind(used_resources_);
  util_->Add(servable_resources, &conservative_proposed_used_resources);

  if (util_->LessThanOrEqual(conservative_proposed_used_resources,
                             total_resources_)) {
    util_->Add(servable_resources, &used_resources_);
    DCHECK(util_->IsNormalized(used_resources_));
    *success = true;
  } else {
    LOG(INFO) << "Insufficient resources to load servable "
              << "\ntotal resources:\n"
              << total_resources_.DebugString() << "used/reserved resources:\n"
              << used_resources_.DebugString()
              << "resources requested by servable:\n"
              << servable_resources.DebugString();
    *success = false;
  }

  return Status::OK();
}

Status ResourceTracker::RecomputeUsedResources(
    const std::vector<const Loader*>& servables) {
  used_resources_.Clear();
  for (const Loader* servable : servables) {
    ResourceAllocation servable_resources;
    TF_RETURN_IF_ERROR(servable->EstimateResources(&servable_resources));
    TF_RETURN_IF_ERROR(util_->VerifyValidity(servable_resources));
    servable_resources = util_->Normalize(servable_resources);
    util_->Add(servable_resources, &used_resources_);
  }
  DCHECK(util_->IsNormalized(used_resources_));
  return Status::OK();
}

ResourceTracker::ResourceTracker(const ResourceAllocation& total_resources,
                                 std::unique_ptr<ResourceUtil> util)
    : util_(std::move(util)), total_resources_(total_resources) {}

}  // namespace serving
}  // namespace tensorflow
