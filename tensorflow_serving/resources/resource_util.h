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

#ifndef TENSORFLOW_SERVING_RESOURCES_RESOURCE_UTIL_H_
#define TENSORFLOW_SERVING_RESOURCES_RESOURCE_UTIL_H_

#include <map>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/resources/resources.pb.h"

namespace tensorflow {
namespace serving {

// Arithmetic and comparison operations on resource allocations.
//
// The implementations assume that the number of devices, and the number of
// instances of each device, are both quite small (fewer than, say, 10). Their
// computational complexity in these dimensions leaves room for improvement.
class ResourceUtil {
 public:
  struct Options {
    // The devices managed by the system, and the number of instances of each.
    std::map<string, uint32> devices;
  };
  explicit ResourceUtil(const Options& options);
  ~ResourceUtil() = default;

  // Determines whether 'allocation' is valid, i.e.:
  //  1. It only refers to valid devices, i.e. those supplied via Options.
  //  2. Each entry is either unbound, or bound to a valid device instance.
  //  3. No distinct Resource entry occurs twice, i.e. resource is a key.
  //
  // All other methods in this class assume their inputs are valid (i.e. they
  // have undefined behavior otherwise), and guarantee to produce valid outputs.
  Status VerifyValidity(const ResourceAllocation& allocation) const;

  // Converts 'allocation' to normal form, meaning:
  //  1. It has no entries with quantity 0.
  //  2. Resources of a device that has exactly one instance are bound to that
  //     instance.
  ResourceAllocation Normalize(const ResourceAllocation& allocation) const;

  // Determines whether 'allocation' is in normal form, as defined above.
  bool IsNormalized(const ResourceAllocation& allocation) const;

  // Determines whether 'allocation' is bound, defined as follows:
  //  1. An individual entry is bound iff a device_instance is supplied.
  //  2. An allocation is bound iff every entry is bound.
  //
  // Assumes the input is normalized.
  bool IsBound(const ResourceAllocation& allocation) const;

  // Adds 'to_add' to 'base'.
  //
  // Keeps bound and unbound entries separate. For example, adding
  // {(GPU/<no_instance>/RAM/8)} to {(GPU/instance_0/RAM/16),
  // (GPU/<no_instance>/RAM/4)} yields {(GPU/instance_0/RAM/16),
  // (GPU/<no_instance>/RAM/12)}.
  //
  // Assumes the inputs are normalized, and produces normalized output.
  void Add(const ResourceAllocation& to_add, ResourceAllocation* base) const;

  // Attempts to subtract 'to_subtract' from 'base'. Like Add(), keeps bound and
  // unbound entries separate. Returns true and mutates 'base' iff the
  // subtraction is legal, i.e. no negative quantities (which cannot be
  // represented) are produced.
  //
  // Assumes the inputs are normalized, and produces normalized output.
  bool Subtract(const ResourceAllocation& to_subtract,
                ResourceAllocation* base) const;

  // Takes a (bound or unbound) allocation 'lhs' and a *bound* allocation 'rhs'.
  // Returns true iff for each entry in 'lhs', either:
  //  1. The entry is bound and its quantity is <= the corresponding one in
  //     'rhs'.
  //  2. The entry is unbound, and there exists an instance I of the device s.t.
  //     the unbound quantity in 'lhs' is <= the quantity in 'rhs' bound to I.
  //
  // IMPORTANT: Assumes 'rhs' is bound; has undefined behavior otherwise.
  //
  // Assumes the inputs are normalized.
  bool LessThanOrEqual(const ResourceAllocation& lhs,
                       const ResourceAllocation& rhs) const;

  // Converts a (potentially) unbound allocation into a bound one, by taking
  // each unbound quantity and binding it to every instance of the device.
  // (Existing bound quantities are preserved.)
  //
  // For example, if there is one CPU and two GPUs then overbinding
  // {(CPU/instance_0/RAM/16), (GPU/<no_instance>/RAM/4)} yields
  // {(CPU/instance_0/RAM/16), (GPU/instance_0/RAM/4), (GPU/instance_1/RAM/4)}.
  //
  // This operation is useful for reasoning about monotonicity and availability
  // of resources, not as a means to permanently bind resources to devices
  // (because it binds resources redundantly to all device instances).
  //
  // Assumes the inputs are normalized, and produces normalized output.
  ResourceAllocation Overbind(const ResourceAllocation& allocation) const;

 private:
  const std::map<string, uint32> devices_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceUtil);
};

// Determines whether two Resource protos are equal.
bool operator==(const Resource& a, const Resource& b);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_UTIL_H_
