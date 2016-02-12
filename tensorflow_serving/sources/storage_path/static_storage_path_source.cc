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

#include "tensorflow_serving/sources/storage_path/static_storage_path_source.h"

#include <functional>
#include <string>

#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

Status StaticStoragePathSource::Create(
    const StaticStoragePathSourceConfig& config,
    std::unique_ptr<StaticStoragePathSource>* result) {
  auto raw_result = new StaticStoragePathSource;
  raw_result->config_ = config;
  result->reset(raw_result);
  return Status::OK();
}

void StaticStoragePathSource::SetAspiredVersionsCallback(
    AspiredVersionsCallback callback) {
  const ServableId id = {config_.servable_name(), config_.version_num()};
  LOG(INFO) << "Aspiring servable " << id;
  callback(config_.servable_name(),
           {CreateServableData(id, config_.version_path())});
}

}  // namespace serving
}  // namespace tensorflow
