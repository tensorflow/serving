/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/core/resource_preserving_policy.h"

namespace tensorflow {
namespace serving {

optional<AspiredVersionPolicy::ServableAction>
ResourcePreservingPolicy::GetNextAction(
    const std::vector<AspiredServableStateSnapshot>& all_versions) const {
  // First iterate over all_versions and find any in kReady that are no longer
  // aspired. Unload the first if any.
  for (const auto& version : all_versions) {
    if (version.state == LoaderHarness::State::kReady && !version.is_aspired) {
      VLOG(1) << "ResourcePreservingPolicy requesting to unload servable "
              << version.id;
      return {{Action::kUnload, version.id}};
    }
  }

  // Second, see if there are any not-aspired versions that aren't in an end
  // state (kDisabled or kError). If so, do nothing for now.
  const bool not_aspired_not_finished =
      std::any_of(all_versions.begin(), all_versions.end(),
                  [](const AspiredServableStateSnapshot& version) {
                    return !version.is_aspired &&
                           version.state != LoaderHarness::State::kDisabled &&
                           version.state != LoaderHarness::State::kError;
                  });
  if (not_aspired_not_finished) {
    return nullopt;
  }

  // Third and only if no action was found earlier, load the aspired new
  // servable with the highest version if any.
  optional<ServableId> highest_aspired_new_version_id =
      GetHighestAspiredNewServableId(all_versions);
  if (highest_aspired_new_version_id) {
    VLOG(1) << "ResourcePreservingPolicy requesting to load servable "
            << highest_aspired_new_version_id;
    return {{Action::kLoad, highest_aspired_new_version_id.value()}};
  }

  return nullopt;
}

}  // namespace serving
}  // namespace tensorflow
