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

#ifndef TENSORFLOW_SERVING_CORE_AVAILABILITY_PRESERVING_POLICY_H_
#define TENSORFLOW_SERVING_CORE_AVAILABILITY_PRESERVING_POLICY_H_

#include <vector>

#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// AspiredVersionPolicy that provides servable availability with the trade-off
// of temporary increased resource consumption while newly-aspired versions load
// followed by newly-un-aspired versions unloading. At the same time, it tries
// to minimize the resource usage caused by loading more versions than needed to
// maintain availability.
//
// Here is a detailed description of how this policy works:
// First, if there are any unaspired loaded versions, we unload the smallest
// such version, *unless* that is the only loaded version (to avoid compromising
// availability).
// Second, if there are no non-aspired versions we are permitted to unload, we
// load the aspired new version with the highest version number.
class AvailabilityPreservingPolicy final : public AspiredVersionPolicy {
 public:
  optional<ServableAction> GetNextAction(
      const std::vector<AspiredServableStateSnapshot>& all_versions)
      const override;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_AVAILABILITY_PRESERVING_POLICY_H_
