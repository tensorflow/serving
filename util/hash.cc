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

#include "tensorflow_serving/util/hash.h"

namespace tensorflow {
namespace serving {

uint64 HashCombine(const uint64 hash1, const uint64 hash2) {
  return hash1 ^ (hash2 + 0x9e3779b97f4a7800 + (hash1 << 10) + (hash1 >> 4));
}

}  // namespace serving
}  // namespace tensorflow
