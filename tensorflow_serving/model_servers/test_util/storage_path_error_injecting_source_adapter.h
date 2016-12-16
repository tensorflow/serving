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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_STORAGE_PATH_ERROR_INJECTING_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_STORAGE_PATH_ERROR_INJECTING_SOURCE_ADAPTER_H_

#include "tensorflow_serving/core/source_adapter.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// An ErrorInjectingSourceAdapter<StoragePath, std::unique_ptr<Loader>> (see
// source_adapter.h) registered in StoragePathSourceAdapterRegistry and keyed on
// StoragePathErrorInjectingSourceAdapterConfig.
using StoragePathErrorInjectingSourceAdapter =
    ErrorInjectingSourceAdapter<StoragePath, std::unique_ptr<Loader>>;

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_TEST_UTIL_STORAGE_PATH_ERROR_INJECTING_SOURCE_ADAPTER_H_
