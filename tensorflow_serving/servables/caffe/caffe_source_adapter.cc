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

#include "tensorflow_serving/servables/caffe/caffe_source_adapter.h"

#include <stddef.h>
#include <memory>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"

namespace tensorflow {
namespace serving {

Status CaffeSourceAdapter::Create(
    const CaffeSourceAdapterConfig& config,
    std::unique_ptr<CaffeSourceAdapter>* adapter) {
  std::unique_ptr<CaffeSessionBundleFactory> bundle_factory;
  TF_RETURN_IF_ERROR(
      CaffeSessionBundleFactory::Create(config.config(), &bundle_factory));
  adapter->reset(new CaffeSourceAdapter(std::move(bundle_factory)));
  return Status::OK();
}

CaffeSourceAdapter::CaffeSourceAdapter(
    std::unique_ptr<CaffeSessionBundleFactory> bundle_factory)
    : bundle_factory_(std::move(bundle_factory)) {}

Status CaffeSourceAdapter::Convert(const StoragePath& path,
                                   std::unique_ptr<Loader>* loader) {
  auto servable_creator = [this, path](std::unique_ptr<CaffeSessionBundle>* bundle) {
    return this->bundle_factory_->CreateSessionBundle(path, bundle);
  };
  auto resource_estimator = [this, path](ResourceAllocation* estimate) {
    return this->bundle_factory_->EstimateResourceRequirement(path, estimate);
  };
  loader->reset(new SimpleLoader<CaffeSessionBundle>(
      servable_creator, resource_estimator));
  return Status::OK();
}

CaffeSourceAdapter::~CaffeSourceAdapter() { Detach(); }

std::function<Status(
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
CaffeSourceAdapter::GetCreator(
    const CaffeSourceAdapterConfig& config) {
  return [&config](std::unique_ptr<tensorflow::serving::SourceAdapter<
                       StoragePath, std::unique_ptr<Loader>>>* source) {
    std::unique_ptr<CaffeSourceAdapter> typed_source;
    TF_RETURN_IF_ERROR(
        CaffeSourceAdapter::Create(config, &typed_source));
    *source = std::move(typed_source);
    return Status::OK();
  };
}


}  // namespace serving
}  // namespace tensorflow
