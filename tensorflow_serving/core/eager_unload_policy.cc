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

#include "tensorflow_serving/core/eager_unload_policy.h"

namespace tensorflow {
namespace serving {

optional<VersionPolicy::ServableAction> EagerUnloadPolicy::GetNextAction(
    const std::vector<ServableStateSnapshot>& all_versions) const {
  // First iterate over all_versions and find any in kReady that are no longer
  // aspired. Unload the first if any.
  for (const auto& version : all_versions) {
    if (version.state == LoaderHarness::State::kReady && !version.is_aspired) {
      return VersionPolicy::ServableAction({Action::kUnload, version.id});
    }
  }

  // Second and only if no action was found earlier, iterate over all
  // versions and find any in kNew that are aspired. Load the first if any.
  for (const auto& version : all_versions) {
    if (version.state == LoaderHarness::State::kNew && version.is_aspired) {
      return VersionPolicy::ServableAction({Action::kLoad, version.id});
    }
  }

  return nullopt;
}

}  // namespace serving
}  // namespace tensorflow
