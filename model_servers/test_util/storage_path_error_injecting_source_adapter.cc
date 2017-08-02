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

#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/model_servers/test_util/storage_path_error_injecting_source_adapter.pb.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Register the source adapter.
class StoragePathErrorInjectingSourceAdapterCreator {
 public:
  static Status Create(
      const StoragePathErrorInjectingSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    adapter->reset(
        new ErrorInjectingSourceAdapter<StoragePath, std::unique_ptr<Loader>>(
            Status(error::CANCELLED, config.error_message())));
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(
    StoragePathErrorInjectingSourceAdapterCreator,
    StoragePathErrorInjectingSourceAdapterConfig);

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
