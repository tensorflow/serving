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

#include "tensorflow_serving/resources/resource_util.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

namespace {

// Obtains the quantity associated with 'resource' in 'allocation'. If none is
// found, returns 0.
uint64 GetQuantityForResource(const Resource& resource,
                              const ResourceAllocation& allocation) {
  for (const ResourceAllocation::Entry& entry :
       allocation.resource_quantities()) {
    if (entry.resource() == resource) {
      return entry.quantity();
    }
  }
  return 0;
}

// Returns a pointer to the entry associated with 'resource' in 'allocation'. If
// none is found, returns nullptr.
ResourceAllocation::Entry* FindMutableEntry(const Resource& resource,
                                            ResourceAllocation* allocation) {
  for (ResourceAllocation::Entry& entry :
       *allocation->mutable_resource_quantities()) {
    if (entry.resource() == resource) {
      return &entry;
    }
  }
  return nullptr;
}

// Returns a pointer to the entry associated with 'resource' in 'allocation'. If
// none is found, inserts an entry with quantity 0 and returns a pointer to it.
ResourceAllocation::Entry* FindOrInsertMutableEntry(
    const Resource& resource, ResourceAllocation* allocation) {
  ResourceAllocation::Entry* entry = FindMutableEntry(resource, allocation);
  if (entry == nullptr) {
    entry = allocation->add_resource_quantities();
    *entry->mutable_resource() = resource;
    entry->set_quantity(0);
  }
  return entry;
}

}  // namespace

ResourceUtil::ResourceUtil(const Options& options) : options_(options) {}

void ResourceUtil::Add(const ResourceAllocation& to_add,
                       ResourceAllocation* base) const {
  for (const ResourceAllocation::Entry& to_add_entry :
       to_add.resource_quantities()) {
    ResourceAllocation::Entry* base_entry =
        FindOrInsertMutableEntry(to_add_entry.resource(), base);
    base_entry->set_quantity(base_entry->quantity() + to_add_entry.quantity());
  }
}

bool ResourceUtil::Subtract(const ResourceAllocation& to_subtract,
                            ResourceAllocation* base) const {
  // We buffer the mutations to 'base' so that if we bail out due to a negative
  // quantity we leave it untouched.
  std::vector<std::pair<ResourceAllocation::Entry*, uint64>> new_quantities;
  for (const ResourceAllocation::Entry& to_subtract_entry :
       to_subtract.resource_quantities()) {
    if (to_subtract_entry.quantity() == 0) {
      continue;
    }
    ResourceAllocation::Entry* base_entry =
        FindMutableEntry(to_subtract_entry.resource(), base);
    if (base_entry == nullptr ||
        base_entry->quantity() < to_subtract_entry.quantity()) {
      LOG(ERROR) << "Subtracting\n"
                 << to_subtract.DebugString() << "from\n"
                 << base->DebugString()
                 << "would result in a negative quantity";
      return false;
    }
    const uint64 new_quantity =
        base_entry->quantity() - to_subtract_entry.quantity();
    new_quantities.push_back({base_entry, new_quantity});
  }
  for (const auto& new_quantity : new_quantities) {
    ResourceAllocation::Entry* base_entry = new_quantity.first;
    const uint64 quantity = new_quantity.second;
    base_entry->set_quantity(quantity);
  }
  return true;
}

bool ResourceUtil::LessThanOrEqual(const ResourceAllocation& a,
                                   const ResourceAllocation& b) const {
  for (const ResourceAllocation::Entry& a_entry : a.resource_quantities()) {
    if (a_entry.quantity() > GetQuantityForResource(a_entry.resource(), b)) {
      return false;
    }
  }
  return true;
}

bool operator==(const Resource& a, const Resource& b) {
  if (a.device() != b.device()) {
    return false;
  }

  if (a.has_device_instance() != b.has_device_instance()) {
    return false;
  }
  if (a.has_device_instance()) {
    if (a.device_instance().value() != b.device_instance().value()) {
      return false;
    }
  }

  return a.kind() == b.kind();
}

}  // namespace serving
}  // namespace tensorflow
