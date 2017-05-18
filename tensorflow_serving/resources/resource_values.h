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

// Standard values to describe resources, to be used for the string types in
// resources.proto. These values should be used whenever applicable, to avoid
// vocabulary mismatches.

#ifndef TENSORFLOW_SERVING_RESOURCES_RESOURCE_VALUES_H_
#define TENSORFLOW_SERVING_RESOURCES_RESOURCE_VALUES_H_

namespace tensorflow {
namespace serving {

// Standard device types.
namespace device_types {

// The primary devices such as CPU(s) and main memory, as well as aspects of the
// server as a whole.
extern const char* const kMain;

// Graphics processing unit(s).
extern const char* const kGpu;

}  // namespace device_types

// Standard resource kinds.
namespace resource_kinds {

// If a server can accommodate at most N models, depicted as the server having N
// "model slots", this is the number of slots needed or allocated.
extern const char* const kNumModelSlots;

// RAM in bytes.
extern const char* const kRamBytes;

// Fraction of a processing unit's cycles, in thousandths.
extern const char* const kProcessingMillis;

// 90th percentile request processing latency, measured in microseconds.
extern const char* const kP90LatencyMicros;

}  // namespace resource_kinds

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_VALUES_H_
