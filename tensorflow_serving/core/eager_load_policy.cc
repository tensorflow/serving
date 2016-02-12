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

namespace tensorflow {
namespace serving {

optional<VersionPolicy::ServableAction> EagerLoadPolicy::GetNextAction(
    const std::vector<ServableStateSnapshot>& all_versions) const {
  // If there is a new aspired version, load it.
  for (const auto& version : all_versions) {
    if (version.is_aspired && version.state == LoaderHarness::State::kNew) {
      return {{Action::kLoad, version.id}};
    }
  }

  // If there is no new aspired version, but a not-aspired version, unload the
  // latter.
  for (const auto& version : all_versions) {
    if (!version.is_aspired && version.state == LoaderHarness::State::kReady) {
      return {{Action::kUnload, version.id}};
    }
  }
  return {};
}

}  // namespace serving
}  // namespace tensorflow
