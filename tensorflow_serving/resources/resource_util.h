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

#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/resources/resources.pb.h"

namespace tensorflow {
namespace serving {

// Arithmetic and comparison operations on resource allocations.
//
// The implementations assume the allocations are very small (i.e. 0-10 entries
// each), and have not been optimized to minimize computational complexity.
class ResourceUtil {
 public:
  struct Options {
    // TODO(b/27494084): Take a parameter that specifies which devices are
    // singleton (i.e. only have one instance), and are hence bound by default.
  };
  explicit ResourceUtil(const Options& options);
  ~ResourceUtil() = default;

  // Adds 'to_add' to 'base'.
  void Add(const ResourceAllocation& to_add, ResourceAllocation* base) const;

  // Attempts to subtract 'to_subtract' from 'base'. Returns true and mutates
  // 'base' iff the subtraction is legal, i.e. no negative quantities (which
  // cannot be represented) are produced.
  bool Subtract(const ResourceAllocation& to_subtract,
                ResourceAllocation* base) const;

  // Determines whether allocation 'a' is less than or equal allocation 'b'.
  //
  // IMPORTANT: For now, deals with unbound and bound entries independently.
  // TODO(b/27494084): Reason properly about unbound and bound entries.
  bool LessThanOrEqual(const ResourceAllocation& a,
                       const ResourceAllocation& b) const;

 private:
  const Options options_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceUtil);
};

// Determines whether two Resource protos are equal.
bool operator==(const Resource& a, const Resource& b);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_UTIL_H_
