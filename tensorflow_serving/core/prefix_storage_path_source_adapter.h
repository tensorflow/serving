/* Copyright 2019 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_CORE_PREFIX_STORAGE_PATH_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_CORE_PREFIX_STORAGE_PATH_SOURCE_ADAPTER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter that adds a prefix to StoragePath.
//
// This can be useful for filesystems which wrap other filesystems via namespace
// prefixes, adding additional functionality like buffering for example.
class PrefixStoragePathSourceAdapter final
    : public UnarySourceAdapter<StoragePath, StoragePath> {
 public:
  explicit PrefixStoragePathSourceAdapter(const std::string& prefix);
  ~PrefixStoragePathSourceAdapter() override;

 protected:
  Status Convert(const StoragePath& source, StoragePath* destination) final;

 private:
  const std::string prefix_;

  TF_DISALLOW_COPY_AND_ASSIGN(PrefixStoragePathSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_PREFIX_STORAGE_PATH_SOURCE_ADAPTER_H_
