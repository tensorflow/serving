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

#include "tensorflow_serving/servables/caffe/caffe_session_bundle_factory.h"

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/caffe/caffe_py_util.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"
#include "tensorflow_serving/servables/caffe/caffe_session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

namespace {

CaffeSessionOptions GetSessionOptions(const CaffeSessionBundleConfig& config) {
  CaffeSessionOptions options;
  options.force_cpu_only = config.force_cpu_only();
  options.force_gpu_id = config.force_gpu_id();

  auto transform_shape = [](const TensorShapeProto& in,
                            CaffeSessionOptions::blob_shape* out) {
    std::transform(std::begin(in.dim()), std::end(in.dim()), out->begin(),
                   [](const TensorShapeProto_Dim& dim) { return dim.size(); });
  };

  {  // initial shape
    const auto& is = config.initial_shape();
    if (!is.unknown_rank() && is.dim_size() > 0) {
      options.initial_shape.reset(
          new CaffeSessionOptions::blob_shape(is.dim_size()));
      transform_shape(is, options.initial_shape.get());
    }
  }
  {  // initial named shapes
    const auto& is = config.named_initial_shapes();
    for (const auto& kvp : is) {
      std::pair<string, CaffeSessionOptions::blob_shape> p = std::make_pair(
          kvp.first, CaffeSessionOptions::blob_shape(kvp.second.dim_size()));
      transform_shape(kvp.second, &p.second);
      options.named_initial_shapes.push_back(std::move(p));
    }
  }

  return options;
}

}  // namespace

constexpr double CaffeSessionBundleFactory::kResourceEstimateRAMMultiplier;
constexpr int CaffeSessionBundleFactory::kResourceEstimateRAMPadBytes;

Status CaffeSessionBundleFactory::Create(
    const CaffeSessionBundleConfig& config,
    std::unique_ptr<CaffeSessionBundleFactory>* factory) {
  std::shared_ptr<Batcher> batcher;
  // Populate 'batcher' if batching is configured.
  if (config.has_batching_parameters()) {
    const BatchingParameters& batching_config = config.batching_parameters();

    if (!batching_config.allowed_batch_sizes().empty()) {
      // Verify that the last allowed batch size matches the max batch size.
      const int last_allowed_size = batching_config.allowed_batch_sizes(
          batching_config.allowed_batch_sizes().size() - 1);
      const int max_size = batching_config.has_max_batch_size()
                               ? batching_config.max_batch_size().value()
                               : Batcher::QueueOptions().max_batch_size;
      if (last_allowed_size != max_size) {
        return errors::InvalidArgument(
            "Last entry in allowed_batch_sizes must match max_batch_size; last "
            "entry was ",
            last_allowed_size, "; expected ", max_size);
      }
    }

    Batcher::Options options;
    if (batching_config.has_num_batch_threads()) {
      options.num_batch_threads = batching_config.num_batch_threads().value();
    }
    if (batching_config.has_thread_pool_name()) {
      options.thread_pool_name = batching_config.thread_pool_name().value();
    }
    TF_RETURN_IF_ERROR(Batcher::Create(options, &batcher));
  }
  factory->reset(new CaffeSessionBundleFactory(config, batcher));
  return Status::OK();
}

Status CaffeSessionBundleFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  const string file_path = io::JoinPath(path, kVariablesFilename);
  if (!Env::Default()->FileExists(file_path).ok()) {
    return errors::NotFound("Nonexistent export path: ", file_path);
  }
  uint64 file_size;
  TF_RETURN_IF_ERROR(Env::Default()->GetFileSize(file_path, &file_size));

  const uint64 ram_requirement =
      file_size * kResourceEstimateRAMMultiplier + kResourceEstimateRAMPadBytes;

  ResourceAllocation::Entry* ram_entry = estimate->add_resource_quantities();
  Resource* ram_resource = ram_entry->mutable_resource();
  ram_resource->set_device(device_types::kMain);
  ram_resource->set_kind(resource_kinds::kRamBytes);
  ram_entry->set_quantity(ram_requirement);

  return Status::OK();
}

Status CaffeSessionBundleFactory::CreateSessionBundle(
    const string& path, std::unique_ptr<CaffeSessionBundle>* bundle) {
  // py-caffe initialization
  if (this->config_.enable_py_caffe()) {
    if (!IsPyCaffeAvailable()) {
      return errors::Internal("PyCaffe requested but is unavilable.");
    }
    TF_RETURN_IF_ERROR(EnsurePyCaffeInitialized());
    for (const string& path : this->config_.python_path()) {
      TF_RETURN_IF_ERROR(EnsurePyCaffeSystemPath(path));
    }
  }

  bundle->reset(new CaffeSessionBundle);
  TF_RETURN_IF_ERROR(LoadSessionBundleFromPath(GetSessionOptions(this->config_),
                                               path, bundle->get()));
  if (this->config_.has_batching_parameters()) {
    TF_RETURN_IF_ERROR(this->WrapSessionForBatching(bundle->get()));
  }
  return Status::OK();
}

CaffeSessionBundleFactory::CaffeSessionBundleFactory(
    const CaffeSessionBundleConfig& config,
    std::shared_ptr<Batcher> batch_scheduler)
    : config_(config), batch_scheduler_(batch_scheduler) {}

Status CaffeSessionBundleFactory::WrapSessionForBatching(
    CaffeSessionBundle* bundle) {
  LOG(INFO) << "Wrapping SessionBundle session to perform batch processing";

  if (batch_scheduler_ == nullptr) {
    return errors::Internal("batch_scheduler_ not set");
  }
  if (!config_.has_batching_parameters()) {
    return errors::Internal("No batching parameters");
  }
  const BatchingParameters& batching_config = config_.batching_parameters();

  Batcher::QueueOptions queue_options;
  if (batching_config.has_max_batch_size()) {
    queue_options.max_batch_size = batching_config.max_batch_size().value();
  }
  if (batching_config.has_batch_timeout_micros()) {
    queue_options.batch_timeout_micros =
        batching_config.batch_timeout_micros().value();
  }
  if (batching_config.has_max_enqueued_batches()) {
    queue_options.max_enqueued_batches =
        batching_config.max_enqueued_batches().value();
  }

  BatchingSessionOptions batching_session_options;
  for (int allowed_batch_size : batching_config.allowed_batch_sizes()) {
    batching_session_options.allowed_batch_sizes.push_back(allowed_batch_size);
  }

  auto create_queue = [this, queue_options](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* batch_scheduler) {
    TF_RETURN_IF_ERROR(this->batch_scheduler_->AddQueue(
        queue_options, process_batch_callback, batch_scheduler));
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(
      CreateBatchingSession(batching_session_options, create_queue,
                            std::move(bundle->session), &bundle->session));

  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
