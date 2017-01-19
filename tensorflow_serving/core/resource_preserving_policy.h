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

#ifndef TENSORFLOW_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_
#define TENSORFLOW_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_

#include <vector>

#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// ServablePolicy that eagerly unloads any no-longer-aspired versions of a
// servable stream and only after done unloading, loads newly aspired versions
// in the order of descending version number.
//
// This policy minimizes resource consumption with the trade-off of temporary
// servable unavailability while all old versions unload followed by the new
// versions loading.
//
// Servables with a single version consuming the majority of their host's
// resources must use this policy to prevent deadlock. Other typical use-cases
// will be for multi-servable environments where clients can tolerate brief
// interruptions to a single servable's availability on a replica.
//
// NB: This policy does not in any way solve cross-replica availability.
class ResourcePreservingPolicy final : public AspiredVersionPolicy {
 public:
  optional<ServableAction> GetNextAction(
      const std::vector<AspiredServableStateSnapshot>& all_versions)
      const override;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_RESOURCE_PRESERVING_POLICY_H_
