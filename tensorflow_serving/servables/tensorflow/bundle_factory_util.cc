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
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"

namespace tensorflow {
namespace serving {

namespace {

using Batcher = SharedBatchScheduler<BatchingSessionTask>;

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

Status EstimateResourceFromPath(const string& path, bool use_validation_result,
                                ResourceAllocation* estimate) {
  TensorflowFileProbingEnv env(Env::Default());
  return EstimateMainRamBytesFromPath(path, use_validation_result, &env,
                                      estimate);
}

Status WrapSessionForBatching(const BatchingParameters& batching_config,
                              std::shared_ptr<Batcher> batch_scheduler,
                              const std::vector<SignatureDef>& signatures,
                              std::unique_ptr<Session>* session,
                              bool enable_default_schedule_creator) {
  LOG(INFO) << "Wrapping session to perform batch processing";

  if (batch_scheduler == nullptr) {
    return errors::Internal("batch_scheduler not set");
  }
  if (*session == nullptr) {
    return errors::Internal("session not set");
  }

  auto queue_options = GetQueueOptions<
      tensorflow::serving::BatchingSessionTask>(
      batching_config,
      [](std::unique_ptr<tensorflow::serving::BatchingSessionTask>* input_task,
         int open_batch_remaining_slot, int max_batch_size,
         std::vector<std::unique_ptr<tensorflow::serving::BatchingSessionTask>>*
             output_tasks) -> tensorflow::Status {
        return SplitInputTask(input_task, open_batch_remaining_slot,
                              max_batch_size, output_tasks);
      });

  BatchingSessionOptions batching_session_options;
  for (int allowed_batch_size : batching_config.allowed_batch_sizes()) {
    batching_session_options.allowed_batch_sizes.push_back(allowed_batch_size);
  }

  batching_session_options.pad_variable_length_inputs =
      batching_config.pad_variable_length_inputs();

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

  // TODO(b/184973097): Remove enable_default_schedule_creator once TFLite is
  // fixed.
  if (enable_default_schedule_creator) {
    return CreateBatchingSession(batching_session_options,
                                 signatures_with_scheduler_creators,
                                 create_queue, std::move(*session), session);
  } else {
    return CreateBatchingSession(batching_session_options,
                                 signatures_with_scheduler_creators,
                                 std::move(*session), session);
  }
}

Status WrapSession(std::unique_ptr<Session>* session) {
  session->reset(new ServingSessionWrapper(std::move(*session)));
  return Status::OK();
}

Status WrapSessionIgnoreThreadPoolOptions(std::unique_ptr<Session>* session) {
  session->reset(
      new SessionWrapperIgnoreThreadPoolOptions(std::move(*session)));
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
