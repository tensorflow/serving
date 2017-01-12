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

#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"

#include <memory>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"

namespace tensorflow {
namespace serving {
namespace test_util {

FakeLoaderSourceAdapter::FakeLoaderSourceAdapter(
    const string& suffix, std::function<void(const string&)> call_on_destruct)
    : SimpleLoaderSourceAdapter(
          [this](const StoragePath& path,
                 std::unique_ptr<string>* servable_ptr) {
            const string servable = suffix_.length() > 0
                                        ? strings::StrCat(path, "/", suffix_)
                                        : path;
            servable_ptr->reset(new string(servable));
            return Status::OK();
          },
          SimpleLoaderSourceAdapter<StoragePath,
                                    string>::EstimateNoResources()),
      suffix_(suffix),
      call_on_destruct_(call_on_destruct) {}

FakeLoaderSourceAdapter::~FakeLoaderSourceAdapter() {
  Detach();
  if (call_on_destruct_) {
    call_on_destruct_(suffix_);
  }
}

// Register the source adapter.
class FakeLoaderSourceAdapterCreator {
 public:
  static Status Create(
      const FakeLoaderSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    adapter->reset(new FakeLoaderSourceAdapter(config.suffix()));
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(FakeLoaderSourceAdapterCreator,
                                     FakeLoaderSourceAdapterConfig);

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
