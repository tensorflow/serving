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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

namespace {

// Returns a copy of 'devices', stripped of any entries whose value is 0.
std::map<string, uint32> StripDevicesWithZeroInstances(
    const std::map<string, uint32>& devices) {
  std::map<string, uint32> result;
  for (const auto& entry : devices) {
    if (entry.second > 0) {
      result.insert(entry);
    }
  }
  return result;
}

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

ResourceUtil::ResourceUtil(const Options& options)
    : devices_(StripDevicesWithZeroInstances(options.devices)) {}

Status ResourceUtil::VerifyValidity(
    const ResourceAllocation& allocation) const {
  // We use 'validated_entries' to look for duplicates.
  ResourceAllocation validated_entries;
  for (const auto& entry : allocation.resource_quantities()) {
    auto it = devices_.find(entry.resource().device());
    if (it == devices_.end()) {
      return errors::InvalidArgument(
          "Invalid resource allocation: Invalid device ",
          entry.resource().device(), " in resource allocation\n",
          allocation.DebugString());
    }
    const uint32 num_instances = it->second;
    if (entry.resource().has_device_instance() &&
        entry.resource().device_instance().value() >= num_instances) {
      return errors::InvalidArgument(
          "Invalid resource allocation: Invalid device instance ",
          entry.resource().device(), ":",
          entry.resource().device_instance().value(),
          " in resource allocation\n", allocation.DebugString());
    }

    if (FindMutableEntry(entry.resource(), &validated_entries) != nullptr) {
      return errors::InvalidArgument(
          "Invalid resource allocation: Repeated resource\n",
          entry.resource().DebugString(), "in allocation\n",
          allocation.DebugString());
    }

    *validated_entries.add_resource_quantities() = entry;
  }
  return Status::OK();
}

ResourceAllocation ResourceUtil::Normalize(
    const ResourceAllocation& allocation) const {
  const Status validity = VerifyValidity(allocation);
  DCHECK_EQ(Status::OK(), validity);
  if (!validity.ok()) {
    LOG(ERROR) << validity;
    return allocation;
  }

  ResourceAllocation normalized;
  for (const auto& entry : allocation.resource_quantities()) {
    if (entry.quantity() == 0) {
      continue;
    }

    ResourceAllocation::Entry* normalized_entry =
        normalized.add_resource_quantities();
    *normalized_entry = entry;
    if (entry.resource().has_device_instance()) {
      continue;
    }
    const uint32 num_instances =
        devices_.find(entry.resource().device())->second;
    if (num_instances == 1) {
      normalized_entry->mutable_resource()
          ->mutable_device_instance()
          ->set_value(0);
    }
  }
  return normalized;
}

bool ResourceUtil::IsNormalized(const ResourceAllocation& allocation) const {
  const Status validity = VerifyValidity(allocation);
  DCHECK_EQ(Status::OK(), validity);
  if (!validity.ok()) {
    LOG(ERROR) << validity;
    return false;
  }

  for (const auto& entry : allocation.resource_quantities()) {
    if (entry.quantity() == 0) {
      return false;
    }

    if (entry.resource().has_device_instance()) {
      continue;
    }
    // For singleton devices (ones that have one instance), the resource should
    // be bound to the single device in the normalized representation.
    const uint32 num_instances =
        devices_.find(entry.resource().device())->second;
    if (num_instances == 1) {
      return false;
    }
  }
  return true;
}

bool ResourceUtil::IsBound(const ResourceAllocation& allocation) const {
  DCHECK(IsNormalized(allocation));
  for (const auto& entry : allocation.resource_quantities()) {
    if (!entry.resource().has_device_instance()) {
      return false;
    }
  }
  return true;
}

void ResourceUtil::Add(const ResourceAllocation& to_add,
                       ResourceAllocation* base) const {
  DCHECK(IsNormalized(to_add));
  DCHECK(IsNormalized(*base));
  for (const ResourceAllocation::Entry& to_add_entry :
       to_add.resource_quantities()) {
    ResourceAllocation::Entry* base_entry =
        FindOrInsertMutableEntry(to_add_entry.resource(), base);
    base_entry->set_quantity(base_entry->quantity() + to_add_entry.quantity());
  }
  DCHECK(IsNormalized(*base));
}

bool ResourceUtil::Subtract(const ResourceAllocation& to_subtract,
                            ResourceAllocation* base) const {
  DCHECK(IsNormalized(to_subtract));
  DCHECK(IsNormalized(*base));
  // We buffer the mutations to 'base' so that if we bail out due to a negative
  // quantity we leave it untouched.
  std::vector<std::pair<ResourceAllocation::Entry*, uint64>> new_quantities;
  for (const ResourceAllocation::Entry& to_subtract_entry :
       to_subtract.resource_quantities()) {
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
  *base = Normalize(*base);
  return true;
}

bool ResourceUtil::LessThanOrEqual(const ResourceAllocation& lhs,
                                   const ResourceAllocation& rhs) const {
  const Status validity = VerifyValidity(lhs);
  DCHECK_EQ(Status::OK(), validity);
  if (!validity.ok()) {
    LOG(ERROR) << validity;
    return false;
  }
  DCHECK(IsNormalized(lhs));
  DCHECK(IsNormalized(rhs));
  DCHECK(IsBound(rhs))
      << "LessThanOrEqual() requires the second argument to be bound";

  // Phase 1: Attempt to subtract the bound entries in 'lhs' from 'rhs'.
  ResourceAllocation subtracted_rhs = rhs;
  for (const ResourceAllocation::Entry& lhs_entry : lhs.resource_quantities()) {
    if (lhs_entry.resource().has_device_instance()) {
      ResourceAllocation to_subtract;
      *to_subtract.add_resource_quantities() = lhs_entry;
      if (!Subtract(to_subtract, &subtracted_rhs)) {
        return false;
      }
    }
  }

  // Phase 2: See if each unbound entry in 'lhs' can fit into a 'subtracted_rhs'
  // via some device instance.
  for (const ResourceAllocation::Entry& lhs_entry : lhs.resource_quantities()) {
    if (!lhs_entry.resource().has_device_instance()) {
      const uint32 num_instances =
          devices_.find(lhs_entry.resource().device())->second;
      Resource bound_resource = lhs_entry.resource();
      bool found_room = false;
      for (int instance = 0; instance < num_instances; ++instance) {
        bound_resource.mutable_device_instance()->set_value(instance);
        if (lhs_entry.quantity() <=
            GetQuantityForResource(bound_resource, subtracted_rhs)) {
          found_room = true;
          break;
        }
      }
      if (!found_room) {
        return false;
      }
    }
  }
  return true;
}

ResourceAllocation ResourceUtil::Overbind(
    const ResourceAllocation& allocation) const {
  const Status validity = VerifyValidity(allocation);
  DCHECK_EQ(Status::OK(), validity);
  if (!validity.ok()) {
    LOG(ERROR) << validity;
    return allocation;
  }
  DCHECK(IsNormalized(allocation));

  ResourceAllocation result;
  for (const ResourceAllocation::Entry& entry :
       allocation.resource_quantities()) {
    if (entry.resource().has_device_instance()) {
      ResourceAllocation::Entry* result_entry =
          FindOrInsertMutableEntry(entry.resource(), &result);
      result_entry->set_quantity(entry.quantity() + result_entry->quantity());
      continue;
    }

    const uint32 num_instances =
        devices_.find(entry.resource().device())->second;
    Resource bound_resource = entry.resource();
    for (uint32 instance = 0; instance < num_instances; ++instance) {
      bound_resource.mutable_device_instance()->set_value(instance);
      ResourceAllocation::Entry* result_entry =
          FindOrInsertMutableEntry(bound_resource, &result);
      result_entry->set_quantity(entry.quantity() + result_entry->quantity());
    }
  }
  DCHECK(IsNormalized(result));
  return result;
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
