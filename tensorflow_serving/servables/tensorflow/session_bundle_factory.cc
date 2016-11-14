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

#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"

namespace tensorflow {
namespace serving {

Status SessionBundleFactory::Create(
    const SessionBundleConfig& config,
    std::unique_ptr<SessionBundleFactory>* factory) {
  std::shared_ptr<Batcher> batcher;
  if (config.has_batching_parameters()) {
    TF_RETURN_IF_ERROR(
        CreateBatchScheduler(config.batching_parameters(), &batcher));
  }
  factory->reset(new SessionBundleFactory(config, batcher));
  return Status::OK();
}

Status SessionBundleFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  return EstimateResourceFromPath(path, estimate);
}

Status SessionBundleFactory::CreateSessionBundle(
    const string& path, std::unique_ptr<SessionBundle>* bundle) {
  bundle->reset(new SessionBundle);
  TF_RETURN_IF_ERROR(LoadSessionBundleFromPathUsingRunOptions(
      GetSessionOptions(config_), GetRunOptions(config_), path, bundle->get()));

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

SessionBundleFactory::SessionBundleFactory(
    const SessionBundleConfig& config, std::shared_ptr<Batcher> batch_scheduler)
    : config_(config), batch_scheduler_(batch_scheduler) {}

}  // namespace serving
}  // namespace tensorflow
