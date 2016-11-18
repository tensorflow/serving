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

#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_factory.h"

#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"

namespace tensorflow {
namespace serving {

Status SavedModelBundleFactory::Create(
    const SessionBundleConfig& config,
    std::unique_ptr<SavedModelBundleFactory>* factory) {
  std::shared_ptr<Batcher> batcher;
  if (config.has_batching_parameters()) {
    TF_RETURN_IF_ERROR(
        CreateBatchScheduler(config.batching_parameters(), &batcher));
  }
  factory->reset(new SavedModelBundleFactory(config, batcher));
  return Status::OK();
}

Status SavedModelBundleFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  return EstimateResourceFromPath(path, estimate);
}

Status SavedModelBundleFactory::CreateSavedModelBundle(
    const string& path, std::unique_ptr<SavedModelBundle>* bundle) {
  bundle->reset(new SavedModelBundle);
  TF_RETURN_IF_ERROR(LoadSessionBundleOrSavedModelBundle(
      GetSessionOptions(config_), GetRunOptions(config_), path,
      {kSavedModelTagServe}, bundle->get()));
  if (config_.has_batching_parameters()) {
    LOG(INFO) << "Wrapping session to perform batch processing";
    if (batch_scheduler_ == nullptr) {
      return errors::Internal("batch_scheduler_ not set");
    }
    return WrapSessionForBatching(config_.batching_parameters(),
                                  batch_scheduler_, &(*bundle)->session);
  }
  return WrapSession(&(*bundle)->session);
}

SavedModelBundleFactory::SavedModelBundleFactory(
    const SessionBundleConfig& config, std::shared_ptr<Batcher> batch_scheduler)
    : config_(config), batch_scheduler_(batch_scheduler) {}

}  // namespace serving
}  // namespace tensorflow
