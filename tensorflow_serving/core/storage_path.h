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

// Typedefs and registries pertaining to storage system paths.

#ifndef TENSORFLOW_SERVING_CORE_STORAGE_PATH_H_
#define TENSORFLOW_SERVING_CORE_STORAGE_PATH_H_

#include <algorithm>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

// Strings that represent paths in some storage system.
using StoragePath = string;

inline bool operator==(const ServableData<StoragePath>& a,
                       const ServableData<StoragePath>& b) {
  if (a.id() != b.id()) {
    return false;
  }
  if (a.status().ok() != b.status().ok()) {
    return false;
  }
  if (a.status().ok()) {
    return a.DataOrDie() == b.DataOrDie();
  } else {
    return a.status() == b.status();
  }
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_STORAGE_PATH_H_
