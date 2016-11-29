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

#ifndef TENSORFLOW_SERVING_SERVABLES_CAFFE_CAFFE_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_CAFFE_CAFFE_SOURCE_ADAPTER_H_

#include <string>
#include <unordered_map>

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/caffe/caffe_source_adapter.pb.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"

namespace tensorflow {
namespace serving {

class CaffeSourceAdapter
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  static Status Create(const CaffeSourceAdapterConfig& config,
                       std::unique_ptr<CaffeSourceAdapter>* adapter);

  ~CaffeSourceAdapter() override;

  // Returns a function to create a session bundle source adapter.
  static std::function<Status(
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
  GetCreator(const CaffeSourceAdapterConfig& config);

 private:
  explicit CaffeSourceAdapter(
      std::unique_ptr<CaffeSessionBundleFactory> bundle_factory);

  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  std::unique_ptr<CaffeSessionBundleFactory> bundle_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(CaffeSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_CAFFE_CAFFE_SOURCE_ADAPTER_H_