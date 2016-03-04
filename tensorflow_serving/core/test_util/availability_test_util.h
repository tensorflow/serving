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

// Methods related to the availability of servables, that are useful in writing
// tests. (Not intended for production use.)

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_

#include "tensorflow_serving/core/servable_state_monitor.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Waits until 'monitor' shows that the manager state of 'servable' is one of
// 'states'.
void WaitUntilServableManagerStateIsOneOf(
    const ServableStateMonitor& monitor, const ServableId& servable,
    const std::vector<ServableState::ManagerState>& states);

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_AVAILABILITY_TEST_UTIL_H_
