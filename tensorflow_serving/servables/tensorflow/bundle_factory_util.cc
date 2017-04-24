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

#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/contrib/batching/batch_scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"

namespace tensorflow {
namespace serving {

namespace {

using Batcher = SharedBatchScheduler<BatchingSessionTask>;

// Constants used in the resource estimation heuristic. See the documentation
// on EstimateResourceFromPath().
constexpr double kResourceEstimateRAMMultiplier = 1.2;
constexpr int kResourceEstimateRAMPadBytes = 0;

// Returns all the descendants, both directories and files, recursively under
// 'dirname'. The paths returned are all prefixed with 'dirname'.
Status GetAllDescendants(const string& dirname, FileProbingEnv* env,
                         std::vector<string>* const descendants) {
  descendants->clear();
  // Make sure that dirname exists;
  TF_RETURN_IF_ERROR(env->FileExists(dirname));
  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all immediate children.
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    TF_RETURN_IF_ERROR(env->GetChildren(dir, &children));
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      descendants->push_back(child_path);
      // If the child is a directory add it to the queue.
      if (env->IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      }
    }
  }
  return Status::OK();
}

}  // namespace

SessionOptions GetSessionOptions(const SessionBundleConfig& config) {
  SessionOptions options;
  options.target = config.session_target();
  options.config = config.session_config();
  return options;
}

RunOptions GetRunOptions(const SessionBundleConfig& config) {
  RunOptions run_options;
  if (config.has_session_run_load_threadpool_index()) {
    run_options.set_inter_op_thread_pool(
        config.session_run_load_threadpool_index().value());
  }
  return run_options;
}

Status CreateBatchScheduler(const BatchingParameters& batching_config,
                            std::shared_ptr<Batcher>* batch_scheduler) {
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
  return Batcher::Create(options, batch_scheduler);
}

Status EstimateResourceFromPath(const string& path,
                                ResourceAllocation* estimate) {
  TensorflowFileProbingEnv env(Env::Default());
  return EstimateResourceFromPath(path, &env, estimate);
}

Status EstimateResourceFromPath(const string& path, FileProbingEnv* env,
                                ResourceAllocation* estimate) {
  if (env == nullptr) {
    return errors::Internal("FileProbingEnv not set");
  }

  std::vector<string> descendants;
  TF_RETURN_IF_ERROR(GetAllDescendants(path, env, &descendants));
  uint64 total_file_size = 0;
  for (const string& descendant : descendants) {
    if (!(env->IsDirectory(descendant).ok())) {
      uint64 file_size;
      TF_RETURN_IF_ERROR(env->GetFileSize(descendant, &file_size));
      total_file_size += file_size;
    }
  }
  const uint64 ram_requirement =
      total_file_size * kResourceEstimateRAMMultiplier +
      kResourceEstimateRAMPadBytes;

  ResourceAllocation::Entry* ram_entry = estimate->add_resource_quantities();
  Resource* ram_resource = ram_entry->mutable_resource();
  ram_resource->set_device(device_types::kMain);
  ram_resource->set_kind(resource_kinds::kRamBytes);
  ram_entry->set_quantity(ram_requirement);

  return Status::OK();
}

Status WrapSessionForBatching(const BatchingParameters& batching_config,
                              std::shared_ptr<Batcher> batch_scheduler,
                              const std::vector<SignatureDef>& signatures,
                              std::unique_ptr<Session>* session) {
  LOG(INFO) << "Wrapping session to perform batch processing";

  if (batch_scheduler == nullptr) {
    return errors::Internal("batch_scheduler not set");
  }
  if (*session == nullptr) {
    return errors::Internal("session not set");
  }

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

  auto create_queue = [batch_scheduler, queue_options](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* queue) {
    TF_RETURN_IF_ERROR(batch_scheduler->AddQueue(
        queue_options, process_batch_callback, queue));
    return Status::OK();
  };
  std::vector<SignatureWithBatchingSessionSchedulerCreator>
      signatures_with_scheduler_creators;
  for (const SignatureDef& signature : signatures) {
    const TensorSignature tensor_signature =
        TensorSignatureFromSignatureDef(signature);
    signatures_with_scheduler_creators.push_back(
        {tensor_signature, create_queue});
  }

  return CreateBatchingSession(batching_session_options,
                               signatures_with_scheduler_creators,
                               std::move(*session), session);
}

Status WrapSession(std::unique_ptr<Session>* session) {
  session->reset(new ServingSessionWrapper(std::move(*session)));
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
