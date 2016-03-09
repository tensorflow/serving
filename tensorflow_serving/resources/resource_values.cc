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

#include "tensorflow_serving/resources/resource_values.h"

namespace tensorflow {
namespace serving {

namespace device_types {
const char* const kCPU = "cpu";
const char* const kGPU = "gpu";
}  // namespace device_types

namespace resource_kinds {
const char* const kRAMBytes = "ram_in_bytes";
const char* const kProcessingMillis = "processing_in_millicores";
}  // namespace resource_kinds

}  // namespace serving
}  // namespace tensorflow
