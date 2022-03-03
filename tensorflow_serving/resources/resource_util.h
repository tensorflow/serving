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
  virtual ~ResourceUtil() = default;

  // Determines whether 'allocation' is valid, i.e.:
  //  1. It only refers to valid devices, i.e. those supplied via Options.
  //  2. Each entry is either unbound, or bound to a valid device instance.
  //  3. No distinct Resource entry occurs twice, i.e. resource is a key.
  //
  // All other methods in this class assume their inputs are valid (i.e. they
  // have undefined behavior otherwise), and guarantee to produce valid outputs.
  virtual Status VerifyValidity(const ResourceAllocation& allocation) const;

  // Verifies whether 'resource' is valid, i.e. it only refers to valid devices,
  // i.e. those supplied via Options.
  Status VerifyResourceValidity(const Resource& resource) const;

  // Converts 'allocation' to normal form, meaning:
  //  1. It has no entries with quantity 0.
  //  2. Resources of a device that has exactly one instance are bound to that
  //     instance.
  virtual ResourceAllocation Normalize(
      const ResourceAllocation& allocation) const;

  // Determines whether 'allocation' is in normal form, as defined above.
  virtual bool IsNormalized(const ResourceAllocation& allocation) const;

  // Determines whether 'allocation' is bound, defined as follows:
  //  1. An individual entry is bound iff a device_instance is supplied.
  //  2. An allocation is bound iff every entry is bound.
  bool IsBound(const ResourceAllocation& allocation) const;

  // Creates a bound resource with the given values. For single-instance
  // resources (which is a common case, e.g. main memory) the 'instance'
  // argument can be omitted.
  Resource CreateBoundResource(const string& device, const string& kind,
                               uint32 device_instance = 0) const;

  // Gets the quantity of 'resource' present in 'allocation'. Returns 0 if
  // 'resource' is not mentioned in 'allocation', since unmentioned resources
  // are implicitly zero.
  uint64_t GetQuantity(const Resource& resource,
                       const ResourceAllocation& allocation) const;

  // Sets the quantity of 'resource' to 'quantity' in 'allocation', overwriting
  // any existing quantity.
  void SetQuantity(const Resource& resource, uint64_t quantity,
                   ResourceAllocation* allocation) const;

  // Adds 'to_add' to 'base'.
  //
  // Keeps bound and unbound entries separate. For example, adding
  // {(GPU/<no_instance>/RAM/8)} to {(GPU/instance_0/RAM/16),
  // (GPU/<no_instance>/RAM/4)} yields {(GPU/instance_0/RAM/16),
  // (GPU/<no_instance>/RAM/12)}.
  void Add(const ResourceAllocation& to_add, ResourceAllocation* base) const;

  // Attempts to subtract 'to_subtract' from 'base'. Like Add(), keeps bound and
  // unbound entries separate. Returns true and mutates 'base' iff the
  // subtraction is legal, i.e. no negative quantities (which cannot be
  // represented) are produced.
  bool Subtract(const ResourceAllocation& to_subtract,
                ResourceAllocation* base) const;

  // Multiplies every resource quantity in 'base' by 'multiplier'. Keeps bound
  // and unbound entries separate.
  void Multiply(uint64_t multiplier, ResourceAllocation* base) const;

  // Determines whether two ResourceAllocation objects are identical (modulo
  // normalization).
  bool Equal(const ResourceAllocation& lhs,
             const ResourceAllocation& rhs) const;

  // Determines whether two Resource objects are identical (modulo
  // normalization).
  bool ResourcesEqual(const Resource& lhs, const Resource& rhs) const;

  // Takes a (bound or unbound) allocation 'lhs' and a *bound* allocation 'rhs'.
  // Returns true iff for each entry in 'lhs', either:
  //  1. The entry is bound and its quantity is <= the corresponding one in
  //     'rhs'.
  //  2. The entry is unbound, and there exists an instance I of the device s.t.
  //     the unbound quantity in 'lhs' is <= the quantity in 'rhs' bound to I.
  //
  // IMPORTANT: Assumes 'rhs' is bound; has undefined behavior otherwise.
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
  ResourceAllocation Overbind(const ResourceAllocation& allocation) const;

  // Gets the max of the two ResourceAllocations by taking the max of every
  // paired entry and keep all the unpaired entries in the max. Like the Add()
  // and Subtract() methods, this keeps the bound and unbound resources
  // separate.
  //
  // E.g.
  // Max({(GPU/instance_0/RAM/8),
  // (CPU/instance_0/processing_in_millicores/100)},
  // {(GPU/instance_0/RAM/16), (CPU/<no_instance>/RAM/4)}) returns
  // {(GPU/instance_0/RAM/16), (CPU/instance_0/processing_in_millicores/100),
  // (CPU/<no_instance>/RAM/4)}
  ResourceAllocation Max(const ResourceAllocation& lhs,
                         const ResourceAllocation& rhs) const;

  // Gets the min of the two ResourceAllocations by taking the min of every
  // paired entry and remove all the unpaired entries in the result. Like the
  // Max() method, this keeps the bound and unbound resources separate.
  //
  // E.g.
  // Min(
  // {(GPU/instance_0/RAM/8), (CPU/instance_0/processing_in_millicores/100)},
  // {(GPU/instance_0/RAM/16), (CPU/<no_instance>/RAM/4)}
  // )
  // returns
  // {(GPU/instance_0/RAM/8)}
  ResourceAllocation Min(const ResourceAllocation& lhs,
                         const ResourceAllocation& rhs) const;

  // The implementation of ResourceUtil::Normalize().
  // Converts 'allocation' to normal form, meaning:
  //  1. It has no entries with quantity 0.
  //  2. Resources of a device that has exactly one instance are bound to that
  //     instance.
  ResourceAllocation NormalizeResourceAllocation(
      const ResourceAllocation& allocation) const;

 private:
  enum class DCHECKFailOption { kDoDCHECKFail, kDoNotDCHECKFail };

  // Wraps fn() with the option to DCHECK-fail.
  Status VerifyFunctionInternal(std::function<Status()> fn,
                                DCHECKFailOption dcheck_fail_option) const;

  // Converts 'resource' to normal form, i.e. ensures that if the device has
  // exactly one instance, the resource is bound to that instance.
  Resource NormalizeResource(const Resource& resource) const;

  // Determines whether 'resource' is normalized. Assumes 'resource' is valid.
  bool IsResourceNormalized(const Resource& resource) const;

  // Like IsBound(), but assumes the input is normalized.
  bool IsBoundNormalized(const ResourceAllocation& allocation) const;

  // Like Add(), but assumes the input is normalized and produces normalized
  // output.
  void AddNormalized(const ResourceAllocation& to_add,
                     ResourceAllocation* base) const;

  // Like Subtract(), but assumes the input is normalized and produces
  // normalized output.
  bool SubtractNormalized(const ResourceAllocation& to_subtract,
                          ResourceAllocation* base) const;

  // Like Multiply(), but assumes the input is normalized and produces
  // normalized output.
  void MultiplyNormalized(uint64_t multiplier, ResourceAllocation* base) const;

  // Like Equal(), but assumes the input is normalized.
  bool EqualNormalized(const ResourceAllocation& lhs,
                       const ResourceAllocation& rhs) const;

  // Like ResourcesEqual(), but assumes the input is normalized.
  bool ResourcesEqualNormalized(const Resource& lhs, const Resource& rhs) const;

  // Like LessThanOrEqual(), but assumes the input is normalized.
  bool LessThanOrEqualNormalized(const ResourceAllocation& lhs,
                                 const ResourceAllocation& rhs) const;

  // Like Overbind(), but assumes the input is normalized and produces
  // normalized output.
  ResourceAllocation OverbindNormalized(
      const ResourceAllocation& allocation) const;

  // Like Max(), but assumes the input are normalized and produces a normalized
  // result.
  ResourceAllocation MaxNormalized(const ResourceAllocation& lhs,
                                   const ResourceAllocation& rhs) const;

  // Like Min(), but assumes the input are normalized and produces a normalized
  // result.
  ResourceAllocation MinNormalized(const ResourceAllocation& lhs,
                                   const ResourceAllocation& rhs) const;

  // The implementation of ResourceUtil::IsNormalized().
  bool IsResourceAllocationNormalized(
      const ResourceAllocation& allocation) const;

  const std::map<string, uint32> devices_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceUtil);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_UTIL_H_
