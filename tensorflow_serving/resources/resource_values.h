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
extern const char* const kCPU;
extern const char* const kGPU;
}  // namespace device_types

// Standard resource kinds.
namespace resource_kinds {
extern const char* const kRAMBytes;
extern const char* const kProcessingMillis;
}  // namespace resource_kinds

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_RESOURCES_RESOURCE_VALUES_H_
