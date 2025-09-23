/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_SOURCE_ADAPTER_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_factory.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter that creates TfrtSavedModelServable Loaders from SavedModel
// paths. It keeps a TfrtSavedModelFactory as its state, which may house a batch
// scheduler that is shared across all of the SavedModel it emits.
class TfrtSavedModelSourceAdapter final
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  static Status Create(const TfrtSavedModelSourceAdapterConfig& config,
                       std::unique_ptr<TfrtSavedModelSourceAdapter>* adapter);

  ~TfrtSavedModelSourceAdapter() override;

 private:
  friend class TfrtSavedModelSourceAdapterCreator;

  explicit TfrtSavedModelSourceAdapter(
      std::unique_ptr<TfrtSavedModelFactory> factory);

  SimpleLoader<Servable>::CreatorVariant GetServableCreator(
      std::shared_ptr<TfrtSavedModelFactory> factory,
      const StoragePath& path) const;

  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  // We use a shared ptr to share ownership with Loaders we emit, in case they
  // outlive this object.
  std::shared_ptr<TfrtSavedModelFactory> factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(TfrtSavedModelSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_SOURCE_ADAPTER_H_
