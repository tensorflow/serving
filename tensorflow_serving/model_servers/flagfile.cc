/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/flagfile.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "absl/strings/str_split.h"

namespace tensorflow {
namespace serving {
namespace main {

static std::string ReadFileIntoString(const std::string& file) {
  std::ifstream inf(file);
  if (!inf.is_open()) {
    return "";
  }

  std::stringstream ss;
  ss << inf.rdbuf();

  return ss.str();
}

std::vector<std::string> LoadFlagsFromFile(const std::string& file) {
  std::string content = ReadFileIntoString(file);

  std::vector<std::string> flags = absl::StrSplit(
      content, '\n', absl::SkipEmpty());

  return flags;
}

}  // namespace main
}  // namespace serving
}  // namespace tensorflow
