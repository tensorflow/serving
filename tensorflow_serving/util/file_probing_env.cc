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

namespace tensorflow {
namespace serving {

Status TensorflowFileProbingEnv::FileExists(const string& fname) {
  return env_->FileExists(fname);
}

Status TensorflowFileProbingEnv::GetChildren(const string& dir,
                                             std::vector<string>* children) {
  return env_->GetChildren(dir, children);
}

Status TensorflowFileProbingEnv::IsDirectory(const string& fname) {
  return env_->IsDirectory(fname);
}

Status TensorflowFileProbingEnv::GetFileSize(const string& fname,
                                             uint64* file_size) {
  return env_->GetFileSize(fname, file_size);
}

}  // namespace serving
}  // namespace tensorflow
