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

#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"

#include <memory>
#include <string>

#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

Status SessionBundleSourceAdapter::Create(
    const SessionBundleSourceAdapterConfig& config,
    std::unique_ptr<SessionBundleSourceAdapter>* adapter) {
  std::unique_ptr<SessionBundleFactory> bundle_factory;
  TF_RETURN_IF_ERROR(
      SessionBundleFactory::Create(config.config(), &bundle_factory));
  adapter->reset(new SessionBundleSourceAdapter(std::move(bundle_factory)));
  return Status::OK();
}

SessionBundleSourceAdapter::SessionBundleSourceAdapter(
    std::unique_ptr<SessionBundleFactory> bundle_factory)
    : bundle_factory_(std::move(bundle_factory)) {}

Status SessionBundleSourceAdapter::Convert(const StoragePath& path,
                                           std::unique_ptr<Loader>* loader) {
  auto servable_creator = [this, path](std::unique_ptr<SessionBundle>* bundle) {
    return this->bundle_factory_->CreateSessionBundle(path, bundle);
  };
  auto resource_estimator = [this, path](ResourceAllocation* estimate) {
    return this->bundle_factory_->EstimateResourceRequirement(path, estimate);
  };
  loader->reset(
      new SimpleLoader<SessionBundle>(servable_creator, resource_estimator));
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
