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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_

#include "google/protobuf/wrappers.pb.h"
#include "absl/types/optional.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/resource_estimator.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

// Returns SessionOptions based on the SessionBundleConfig.
// TODO(b/32248363): add SavedModelBundleConfig after we switch Model Server to
// Saved Model.
SessionOptions GetSessionOptions(const SessionBundleConfig& config);

// Returns RunOptions based on SessionBundleConfig.
// TODO(b/32248363): add SavedModelBundleConfig after we switch Model Server to
// Saved Model.
RunOptions GetRunOptions(const SessionBundleConfig& config);

// Get per-model batching parameters if they are present.
//
// When `per_model_configured` is true we return model specific batching
// parameters from `batching_params.pbtxt` file in SavedModel dir under `path`
// if one exists.  If `per_model_configured` is false we return `common_params`.
// Failure to parse model specific params will return error.
Status GetPerModelBatchingParams(const string& path,
                                 const BatchingParameters& common_params,
                                 bool per_model_configured,
                                 absl::optional<BatchingParameters>* params);

// Creates a BatchScheduler based on the batching configuration.
template <typename TaskType>
Status CreateBatchScheduler(
    const BatchingParameters& batching_config,
    std::shared_ptr<SharedBatchScheduler<TaskType>>* batch_scheduler) {
  typename SharedBatchScheduler<TaskType>::Options options;
  if (batching_config.has_num_batch_threads()) {
    options.num_batch_threads = batching_config.num_batch_threads().value();
  }
  if (batching_config.has_thread_pool_name()) {
    options.thread_pool_name = batching_config.thread_pool_name().value();
  }
  return SharedBatchScheduler<TaskType>::Create(options, batch_scheduler);
}

// Estimates the resources a session bundle or saved model bundle will use once
// loaded, from its export or saved model path. tensorflow::Env::Default() will
// be used to access the file system.
//
// If use_validation_result = true, tries to use the result from infra validtion
// first. Otherwise, uses the following crude heuristic: estimated main-memory
// RAM = (combined size of all exported file(s)) *
// kResourceEstimateRAMMultiplier + kResourceEstimateRAMPadBytes.
// TODO(b/27694447): Improve the heuristic. At a minimum, account for GPU RAM.
Status EstimateResourceFromPath(const string& path, bool use_validation_result,
                                ResourceAllocation* estimate);

// Wraps a session in a new session that automatically batches Run() calls.
Status WrapSessionForBatching(
    const BatchingParameters& batching_config,
    std::shared_ptr<SharedBatchScheduler<BatchingSessionTask>> batch_scheduler,
    const std::vector<SignatureDef>& signatures,
    std::unique_ptr<Session>* session);

// Wraps a session in a new session that only supports Run() without batching.
Status WrapSession(std::unique_ptr<Session>* session);

// Wraps a session in a new session that only supports Run() without threading
// parameters.
Status WrapSessionIgnoreThreadPoolOptions(std::unique_ptr<Session>* session);

// Construct Queue Options from BatchingParameters.
template <typename TaskType>
typename SharedBatchScheduler<TaskType>::QueueOptions GetQueueOptions(
    const BatchingParameters& batching_config,
    std::function<Status(std::unique_ptr<TaskType>* input_task,
                         int first_output_task_size, int input_batch_size_limit,
                         std::vector<std::unique_ptr<TaskType>>* output_tasks)>
        split_input_task_func) {
  typename SharedBatchScheduler<TaskType>::QueueOptions queue_options;
  if (batching_config.has_max_batch_size()) {
    queue_options.input_batch_size_limit =
        batching_config.max_batch_size().value();
  }
  if (batching_config.has_batch_timeout_micros()) {
    queue_options.batch_timeout_micros =
        batching_config.batch_timeout_micros().value();
  }
  if (batching_config.has_max_enqueued_batches()) {
    queue_options.max_enqueued_batches =
        batching_config.max_enqueued_batches().value();
  }
  if (batching_config.has_enable_large_batch_splitting() &&
      batching_config.enable_large_batch_splitting().value()) {
    queue_options.enable_large_batch_splitting = true;

    if (batching_config.has_max_execution_batch_size()) {
      queue_options.max_execution_batch_size =
          batching_config.max_execution_batch_size().value();
    } else {
      queue_options.max_execution_batch_size =
          batching_config.max_batch_size().value();
    }

    queue_options.split_input_task_func = split_input_task_func;
  }
  return queue_options;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_
