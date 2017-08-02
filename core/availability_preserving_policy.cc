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

#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/loader_harness.h"

namespace tensorflow {
namespace serving {

namespace {

// Returns the ServableId with the lowest version, if any exists.
optional<ServableId> GetLowestServableId(
    const std::vector<AspiredServableStateSnapshot>& all_versions) {
  const auto& iterator =
      std::min_element(all_versions.begin(), all_versions.end(),
                       [](const AspiredServableStateSnapshot& a,
                          const AspiredServableStateSnapshot& b) {
                         return a.id.version < b.id.version;
                       });
  if (iterator == all_versions.end()) {
    return nullopt;
  } else {
    return iterator->id;
  }
}

}  // namespace

optional<AspiredVersionPolicy::ServableAction>
AvailabilityPreservingPolicy::GetNextAction(
    const std::vector<AspiredServableStateSnapshot>& all_versions) const {
  // We first try to unload non-aspired versions (if any).
  bool has_aspired = false;
  bool has_aspired_serving = false;
  std::vector<AspiredServableStateSnapshot> unaspired_serving_versions;
  for (const auto& version : all_versions) {
    if (version.is_aspired) {
      has_aspired = true;
      if (version.state == LoaderHarness::State::kReady) {
        has_aspired_serving = true;
      }
    } else if (version.state == LoaderHarness::State::kReady) {
      unaspired_serving_versions.push_back(version);
    }
  }

  // If there is no aspired version, there is at least one aspired version
  // that is ready, or there are more than one un-aspired versions that are
  // ready, unload the lowest non-aspired version.
  if (!has_aspired || has_aspired_serving ||
      unaspired_serving_versions.size() > 1) {
    optional<ServableId> version_to_unload =
        GetLowestServableId(unaspired_serving_versions);
    if (version_to_unload) {
      return {{Action::kUnload, version_to_unload.value()}};
    }
  }

  // If there is at least one new aspired version, load the one with the
  // highest version number.
  optional<ServableId> highest_new_aspired_version_id =
      GetHighestAspiredNewServableId(all_versions);
  if (highest_new_aspired_version_id) {
    VLOG(1) << "AvailabilityPreservingPolicy requesting to load servable "
            << highest_new_aspired_version_id.value();
    return {{Action::kLoad, highest_new_aspired_version_id.value()}};
  }

  return nullopt;
}

}  // namespace serving
}  // namespace tensorflow
