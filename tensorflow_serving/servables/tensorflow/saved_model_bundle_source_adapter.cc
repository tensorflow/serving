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

#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.h"

#include <memory>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

Status SavedModelBundleSourceAdapter::Create(
    const SessionBundleSourceAdapterConfig& config,
    std::unique_ptr<SavedModelBundleSourceAdapter>* adapter) {
  std::unique_ptr<SavedModelBundleFactory> bundle_factory;
  TF_RETURN_IF_ERROR(
      SavedModelBundleFactory::Create(config.config(), &bundle_factory));
  adapter->reset(new SavedModelBundleSourceAdapter(std::move(bundle_factory)));
  return Status::OK();
}

SavedModelBundleSourceAdapter::~SavedModelBundleSourceAdapter() { Detach(); }

SavedModelBundleSourceAdapter::SavedModelBundleSourceAdapter(
    std::unique_ptr<SavedModelBundleFactory> bundle_factory)
    : bundle_factory_(std::move(bundle_factory)) {}

Status SavedModelBundleSourceAdapter::Convert(const StoragePath& path,
                                              std::unique_ptr<Loader>* loader) {
  std::shared_ptr<SavedModelBundleFactory> bundle_factory = bundle_factory_;
  auto servable_creator = [bundle_factory,
                           path](std::unique_ptr<SavedModelBundle>* bundle) {
    return bundle_factory->CreateSavedModelBundle(path, bundle);
  };
  auto resource_estimator = [bundle_factory,
                             path](ResourceAllocation* estimate) {
    return bundle_factory->EstimateResourceRequirement(path, estimate);
  };
  loader->reset(
      new SimpleLoader<SavedModelBundle>(servable_creator, resource_estimator));
  return Status::OK();
}

std::function<Status(
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
SavedModelBundleSourceAdapter::GetCreator(
    const SessionBundleSourceAdapterConfig& config) {
  return [config](std::unique_ptr<tensorflow::serving::SourceAdapter<
                      StoragePath, std::unique_ptr<Loader>>>* source) {
    std::unique_ptr<SavedModelBundleSourceAdapter> typed_source;
    TF_RETURN_IF_ERROR(
        SavedModelBundleSourceAdapter::Create(config, &typed_source));
    *source = std::move(typed_source);
    return Status::OK();
  };
}

// Register the source adapter.
class SavedModelBundleSourceAdapterCreator {
 public:
  static Status Create(
      const SavedModelBundleSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    std::unique_ptr<SavedModelBundleFactory> bundle_factory;
    TF_RETURN_IF_ERROR(SavedModelBundleFactory::Create(config.legacy_config(),
                                                       &bundle_factory));
    adapter->reset(
        new SavedModelBundleSourceAdapter(std::move(bundle_factory)));
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(SavedModelBundleSourceAdapterCreator,
                                     SavedModelBundleSourceAdapterConfig);

}  // namespace serving
}  // namespace tensorflow
