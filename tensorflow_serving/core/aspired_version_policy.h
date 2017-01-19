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

#ifndef TENSORFLOW_SERVING_CORE_ASPIRED_VERSION_POLICY_H_
#define TENSORFLOW_SERVING_CORE_ASPIRED_VERSION_POLICY_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// A snapshot of a servable's state and aspiredness.
struct AspiredServableStateSnapshot final {
  ServableId id;
  LoaderHarness::State state;
  bool is_aspired;
};

// An interface for the policy to be applied for transitioning servable versions
// in a servable stream.
//
// Policies should be entirely stateless and idempotent. Asking the same policy
// multiple times for the next action, for an identical vector of
// AspiredServableStateSnapshots, should return the same result.
//
// If additional state is required to implement a Policy, such state shall be
// shared via AspiredServableStateSnapshots. Depending on the kind of state, the
// most likely candidates for originating or tracking state are Sources or the
// Harness and Manager.
class AspiredVersionPolicy {
 public:
  // The different actions that could be recommended by a policy.
  enum class Action : int {
    // Call load on the servable.
    kLoad,
    // Call unload on the servable.
    kUnload,
  };

  virtual ~AspiredVersionPolicy() = default;

  // Action and the id of the servable associated with it.
  struct ServableAction final {
    Action action;
    ServableId id;

    string DebugString() const {
      return strings::StrCat("{ action: ", static_cast<int>(action), " id: ",
                             id.DebugString(), " }");
    }
  };

  // Takes in a vector of state snapshots of all versions of a servable stream
  // and returns an action to be performed for a particular servable version,
  // depending only on the states of all the versions.
  //
  // If no action is to be performed, we don't return an action, meaning
  // that the servable stream is up to date.
  virtual optional<ServableAction> GetNextAction(
      const std::vector<AspiredServableStateSnapshot>& all_versions) const = 0;

 protected:
  // Returns the aspired ServableId with the highest version that matches
  // kNew state, if any exists.
  static optional<ServableId> GetHighestAspiredNewServableId(
      const std::vector<AspiredServableStateSnapshot>& all_versions);

 private:
  friend class AspiredVersionPolicyTest;
};

inline bool operator==(const AspiredVersionPolicy::ServableAction& lhs,
                       const AspiredVersionPolicy::ServableAction& rhs) {
  return lhs.action == rhs.action && lhs.id == rhs.id;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_ASPIRED_VERSION_POLICY_H_
