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

#include "tensorflow_serving/core/eager_load_policy.h"
#include "tensorflow_serving/core/loader_harness.h"

namespace tensorflow {
namespace serving {

optional<AspiredVersionPolicy::ServableAction> EagerLoadPolicy::GetNextAction(
    const std::vector<AspiredServableStateSnapshot>& all_versions) const {
  // If there is a new aspired version, load it.
  for (const auto& version : all_versions) {
    if (version.is_aspired && version.state == LoaderHarness::State::kNew) {
      VLOG(1) << "EagerLoadPolicy requesting to load servable " << version.id;
      return {{Action::kLoad, version.id}};
    }
  }

  // Second, check if there are any aspired versions that are not ready. In that
  // case we can't unload any versions.
  const bool aspired_not_serving =
      std::any_of(all_versions.begin(), all_versions.end(),
                  [](const AspiredServableStateSnapshot& version) {
                    return version.is_aspired &&
                           version.state != LoaderHarness::State::kReady;
                  });
  if (aspired_not_serving) {
    return nullopt;
  }

  // If there is no new aspired version, but a not-aspired version, unload the
  // latter.
  for (const auto& version : all_versions) {
    if (!version.is_aspired && version.state == LoaderHarness::State::kReady) {
      VLOG(1) << "EagerLoadPolicy requesting to unload servable " << version.id;
      return {{Action::kUnload, version.id}};
    }
  }
  return nullopt;
}

}  // namespace serving
}  // namespace tensorflow
