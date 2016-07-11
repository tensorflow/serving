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

#ifndef TENSORFLOW_SERVING_UTIL_HASH_H_
#define TENSORFLOW_SERVING_UTIL_HASH_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// Combines 2 hashes and returns a 3rd one.
uint64 HashCombine(uint64 hash1, uint64 hash2);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_HASH_H_
