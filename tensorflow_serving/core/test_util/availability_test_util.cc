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

#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {
namespace test_util {

namespace {

// Determines whether WaitUntilServableManagerStateIsOneOf()'s condition is
// satisfied. (See that function's documentation.)
bool ServableManagerStateIsOneOf(
    const ServableStateMonitor& monitor, const ServableId& servable,
    const std::vector<ServableState::ManagerState>& states) {
  optional<ServableState> maybe_state = monitor.GetState(servable);
  if (!maybe_state) {
    return false;
  }
  const ServableState state = *maybe_state;

  for (const ServableState::ManagerState& desired_manager_state : states) {
    if (state.manager_state == desired_manager_state) {
      return true;
    }
  }
  return false;
}

}  // namespace

void WaitUntilServableManagerStateIsOneOf(
    const ServableStateMonitor& monitor, const ServableId& servable,
    const std::vector<ServableState::ManagerState>& states) {
  while (!ServableManagerStateIsOneOf(monitor, servable, states)) {
    Env::Default()->SleepForMicroseconds(50 * 1000 /* 50 ms */);
  }
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
