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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_factory.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter that creates SavedModelBundle Loaders from SavedModel paths.
// It keeps a SavedModelBundleFactory as its state, which may house a batch
// scheduler that is shared across all of the SavedModel bundles it emits.
class SavedModelBundleSourceAdapter final
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  // TODO(b/32248363): Switch to SavedModelBundleSourceAdapterConfig after we
  // switch Model Server to Saved Model and populate the "real" fields of
  // SavedModelBundleSourceAdapterConfig.
  static Status Create(const SessionBundleSourceAdapterConfig& config,
                       std::unique_ptr<SavedModelBundleSourceAdapter>* adapter);

  ~SavedModelBundleSourceAdapter() override;

  // Returns a function to create a SavedModel bundle source adapter.
  static std::function<Status(
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
  GetCreator(const SessionBundleSourceAdapterConfig& config);

 private:
  friend class SavedModelBundleSourceAdapterCreator;

  explicit SavedModelBundleSourceAdapter(
      std::unique_ptr<SavedModelBundleFactory> bundle_factory);

  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  // We use a shared ptr to share ownership with Loaders we emit, in case they
  // outlive this object.
  std::shared_ptr<SavedModelBundleFactory> bundle_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelBundleSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_SOURCE_ADAPTER_H_
