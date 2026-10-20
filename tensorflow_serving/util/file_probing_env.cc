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

#include "tensorflow_serving/util/file_probing_env.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace serving {

absl::Status TensorflowFileProbingEnv::FileExists(absl::string_view fname) {
  return env_->FileExists(std::string(fname));
}

absl::Status TensorflowFileProbingEnv::GetChildren(
    const std::string& dir, std::vector<std::string>* children) {
  return env_->GetChildren(dir, children);
}

absl::Status TensorflowFileProbingEnv::IsDirectory(const std::string& fname) {
  return env_->IsDirectory(fname);
}

absl::Status TensorflowFileProbingEnv::GetFileSize(const std::string& fname,
                                                   uint64_t* file_size) {
  return env_->GetFileSize(fname, file_size);
}

}  // namespace serving
}  // namespace tensorflow
