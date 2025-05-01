/* Copyright 2020 Google Inc. All Rights Reserved.
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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_RUN_OPTIONS_BASE_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_RUN_OPTIONS_BASE_H_

#include <stdint.h>

#include "absl/time/time.h"

namespace tensorflow {
namespace serving {
namespace servables {

// RunOptionsBase group the configuration for individual inference executions.
// The per-request configuration (e.g. deadline) can be passed here.
struct RunOptionsBase {
  // Priority of the request. Some thread pool implementation will schedule
  // ops based on the priority number. Larger number means higher
  // priority.
  int64_t priority = 1;

  // The deadline for this request.
  absl::Time deadline = absl::InfiniteFuture();

  // Controls the latency prioritization of a request within a priority.
  // Requests with higher priority always get prioritized for latency over
  // requests with lower priority. 0 is the lowest latency priority.
  int32_t latency_priority = 0;
};

}  // namespace servables
}  // namespace serving
}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_OSS_RUN_OPTIONS_BASE_H_
