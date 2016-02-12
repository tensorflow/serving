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

#ifndef TENSORFLOW_SERVING_CORE_EAGER_LOAD_POLICY_H_
#define TENSORFLOW_SERVING_CORE_EAGER_LOAD_POLICY_H_

#include <vector>

#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/core/version_policy.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// VersionPolicy that loads any aspired versions of a servable before
// unloading any no-longer-aspired versions.
//
// This policy provides servable availability with the trade-off of temporary
// increased resource consumption while the new version loads followed by the
// old versions unloading.
class EagerLoadPolicy final : public VersionPolicy {
 public:
  optional<ServableAction> GetNextAction(
      const std::vector<ServableStateSnapshot>& all_versions) const override;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_EAGER_LOAD_POLICY_H_
