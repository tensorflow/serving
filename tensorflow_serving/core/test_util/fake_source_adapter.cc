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

#include "tensorflow_serving/core/test_util/fake_source_adapter.h"

namespace tensorflow {
namespace serving {
namespace test_util {

FakeSourceAdapter::FakeSourceAdapter()
    : SimpleLoaderSourceAdapter(
          [](const StoragePath& path, std::unique_ptr<string>* servable_ptr) {
            servable_ptr->reset(new string(path));
            return Status::OK();
          },
          SimpleLoaderSourceAdapter<StoragePath,
                                    string>::EstimateNoResources()) {}

std::function<Status(
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
FakeSourceAdapter::GetCreator() {
  return [](std::unique_ptr<tensorflow::serving::SourceAdapter<
                StoragePath, std::unique_ptr<Loader>>>* source) {
    source->reset(new FakeSourceAdapter);
    return Status::OK();
  };
}
}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
