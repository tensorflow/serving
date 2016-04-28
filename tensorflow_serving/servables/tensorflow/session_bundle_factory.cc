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

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

namespace {

SessionOptions GetSessionOptions(const SessionBundleConfig& config) {
  SessionOptions options;
  options.target = config.session_target();
  options.config = config.session_config();
  return options;
}

}  // namespace

constexpr double SessionBundleFactory::kResourceEstimateRAMMultiplier;
constexpr int SessionBundleFactory::kResourceEstimateRAMPadBytes;

Status SessionBundleFactory::Create(
    const SessionBundleConfig& config,
    std::unique_ptr<SessionBundleFactory>* factory) {
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
  factory->reset(new SessionBundleFactory(config, batcher));
  return Status::OK();
}

Status SessionBundleFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  const char kVariablesFilenameRegexp[] = "export(-[0-9]+-of-[0-9]+)?";
  if (!Env::Default()->FileExists(path)) {
    return errors::NotFound("Nonexistent export path: ", path);
  }

  uint64 total_variable_file_size = 0;
  std::vector<string> files;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(path, &files));
  for (const string& file : files) {
    if (!RE2::FullMatch(file, kVariablesFilenameRegexp)) {
      continue;
    }
    const string file_path = io::JoinPath(path, file);
    uint64 file_size;
    TF_RETURN_IF_ERROR(Env::Default()->GetFileSize(file_path, &file_size));
    total_variable_file_size += file_size;
  }
  const uint64 ram_requirement =
      total_variable_file_size * kResourceEstimateRAMMultiplier +
      kResourceEstimateRAMPadBytes;

  ResourceAllocation::Entry* ram_entry = estimate->add_resource_quantities();
  Resource* ram_resource = ram_entry->mutable_resource();
  ram_resource->set_device(device_types::kMain);
  ram_resource->set_kind(resource_kinds::kRamBytes);
  ram_entry->set_quantity(ram_requirement);

  return Status::OK();
}

Status SessionBundleFactory::CreateSessionBundle(
    const string& path, std::unique_ptr<SessionBundle>* bundle) {
  bundle->reset(new SessionBundle);
  TF_RETURN_IF_ERROR(LoadSessionBundleFromPath(GetSessionOptions(this->config_),
                                               path, bundle->get()));
  if (this->config_.has_batching_parameters()) {
    TF_RETURN_IF_ERROR(this->WrapSessionForBatching(bundle->get()));
  } else {
    (*bundle)->session.reset(
        new ServingSessionWrapper(std::move((*bundle)->session)));
  }
  return Status::OK();
}

SessionBundleFactory::SessionBundleFactory(
    const SessionBundleConfig& config, std::shared_ptr<Batcher> batch_scheduler)
    : config_(config), batch_scheduler_(batch_scheduler) {}

Status SessionBundleFactory::WrapSessionForBatching(SessionBundle* bundle) {
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

  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  if (batching_config.has_max_time_micros()) {
    retry_options.max_time_micros = batching_config.max_time_micros().value();
  }
  if (batching_config.has_retry_delay_micros()) {
    retry_options.retry_delay_micros =
        batching_config.retry_delay_micros().value();
  }

  auto create_queue = [this, queue_options, retry_options](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* batch_scheduler) {
    std::unique_ptr<BatchScheduler<BatchingSessionTask>> queue;
    TF_RETURN_IF_ERROR(this->batch_scheduler_->AddQueue(
        queue_options, process_batch_callback, &queue));
    std::unique_ptr<BatchSchedulerRetrier<BatchingSessionTask>> retrier;
    TF_RETURN_IF_ERROR(BatchSchedulerRetrier<BatchingSessionTask>::Create(
        retry_options, std::move(queue), &retrier));
    *batch_scheduler = std::move(retrier);
    return Status::OK();
  };
  TF_RETURN_IF_ERROR(
      CreateBatchingSession(batching_session_options, create_queue,
                            std::move(bundle->session), &bundle->session));

  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
