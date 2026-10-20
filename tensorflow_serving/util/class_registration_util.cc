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

#include "tensorflow_serving/util/class_registration_util.h"

#include <vector>

namespace tensorflow {
namespace serving {

absl::Status ParseUrlForAnyType(const std::string& type_url,
                                std::string* const full_type_name) {
  std::vector<std::string> splits = str_util::Split(type_url, '/');
  if (splits.size() < 2 || splits[splits.size() - 1].empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Supplied config's type_url could not be parsed: ", type_url));
  }
  *full_type_name = splits[splits.size() - 1];
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
