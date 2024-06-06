/* Copyright 2023 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/oss/file_acl.h"

#include <string_view>

#include "absl/status/status.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

absl::Status RegisterModelRoot(const ServableId& servable_id,
                               std::string_view root_path) {
  // Unimplemented
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
