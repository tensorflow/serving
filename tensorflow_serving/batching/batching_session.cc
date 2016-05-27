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

#include "tensorflow_serving/batching/batching_session.h"

#include <stddef.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/batching/batch_scheduler_retrier.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/util/cleanup.h"

namespace tensorflow {
namespace serving {

// A session that performs batching on top of a wrapped session. See the
// documentation in batching_session.h for details and constraints.
class BatchingSession : public ServingSession {
 public:
  // Constructs a BatchingSession. Arguments:
  // - 'options' contains parameters. See batching_session.h.
  // - 'wrapped' is the session to wrap with batching.
  // - 'batch_scheduler_creator' constructs a batch scheduler given a process-
  //   batch callback. See batching_session.h for example usage.
  static Status Create(
      const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
      std::function<Status(
          std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>,
          std::unique_ptr<BatchScheduler<BatchingSessionTask>>*)>
          batch_scheduler_creator,
      std::unique_ptr<BatchingSession>* result);

  ~BatchingSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

 private:
  explicit BatchingSession(const BatchingSessionOptions& options);

  // Computes the size of an input tensor list for batching purposes, by
  // analyzing the 0th dimension size of each of the tensors. All tensors in the
  // list must have the same 0th dimension size to be batchable. If the sizes
  // are not all identical, returns an error.
  Status ComputeInputSize(const std::vector<std::pair<string, Tensor>>& inputs,
                          size_t* size) const;

  // Returns the smallest entry in 'options_.allowed_batch_sizes' that is
  // greater than or equal to 'batch_size'. If 'options_.allowed_batch_sizes' is
  // empty, simply returns 'batch_size'.
  int RoundToLowestAllowedBatchSize(int batch_size) const;

  // Merges the input tensors in a batch, via concatenation of correspondingly-
  // named tensors, and extracts the output tensor names. Assumes 'batch' is
  // non-empty. Returns an error if there are any mismatches among the tasks in
  // the batch that violate the constraints for batchability.
  Status MergeInputTensors(
      const Batch<BatchingSessionTask>& batch,
      std::vector<std::pair<string, Tensor>>* merged_inputs,
      std::vector<string>* output_tensor_names);

  // Splits the output of a batched call to 'wrapped_->Run()' into individual
  // task outputs.
  Status SplitOutputTensors(const std::vector<Tensor>& combined_outputs,
                            Batch<BatchingSessionTask>* batch);

  // Processes one batch. Called by 'batch_scheduler_' in a batch thread.
  void ProcessBatch(std::unique_ptr<Batch<BatchingSessionTask>> batch);

  const BatchingSessionOptions options_;

  std::unique_ptr<Session> wrapped_;
  std::unique_ptr<BatchScheduler<BatchingSessionTask>> batch_scheduler_;

  TF_DISALLOW_COPY_AND_ASSIGN(BatchingSession);
};

Status BatchingSession::Create(
    const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
    std::function<
        Status(std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>,
               std::unique_ptr<BatchScheduler<BatchingSessionTask>>*)>
        batch_scheduler_creator,
    std::unique_ptr<BatchingSession>* result) {
  auto batching_session =
      std::unique_ptr<BatchingSession>(new BatchingSession(options));
  BatchingSession* raw_batching_session = batching_session.get();
  batching_session->wrapped_ = std::move(wrapped);
  std::unique_ptr<BatchScheduler<BatchingSessionTask>> batch_scheduler;
  TF_RETURN_IF_ERROR(batch_scheduler_creator(
      [raw_batching_session](
          std::unique_ptr<Batch<BatchingSessionTask>> batch) {
        raw_batching_session->ProcessBatch(std::move(batch));
      },
      &batch_scheduler));
  batching_session->batch_scheduler_ = std::move(batch_scheduler);
  *result = std::move(batching_session);
  return Status::OK();
}

Status BatchingSession::Run(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs) {
  if (!target_node_names.empty()) {
    return errors::PermissionDenied(
        "BatchingSession does not support target nodes");
  }

  Notification done;
  Status status;
  auto task = std::unique_ptr<BatchingSessionTask>(new BatchingSessionTask);
  TF_RETURN_IF_ERROR(ComputeInputSize(inputs, &task->zeroth_dim_size));
  task->inputs = &inputs;
  task->output_tensor_names = &output_tensor_names;
  task->done = &done;
  task->status = &status;
  task->outputs = outputs;

  TF_RETURN_IF_ERROR(batch_scheduler_->Schedule(&task));
  done.WaitForNotification();
  return status;
}

BatchingSession::BatchingSession(const BatchingSessionOptions& options)
    : options_(options) {}

Status BatchingSession::ComputeInputSize(
    const std::vector<std::pair<string, Tensor>>& inputs, size_t* size) const {
  if (inputs.size() == 0) {
    return errors::InvalidArgument(
        "Batching session Run() must have at least one input tensor");
  }

  bool first = true;
  for (const auto& entry : inputs) {
    const Tensor& tensor = entry.second;

    if (tensor.shape().dims() == 0) {
      return errors::InvalidArgument(
          "Batching session Run() input tensors must have at least one "
          "dimension");
    }
    const size_t this_size = tensor.shape().dim_size(0);

    if (first) {
      *size = this_size;
      first = false;
    } else {
      if (this_size != *size) {
        return errors::InvalidArgument(
            "Batching session Run() input tensors must have equal "
            "0th-dimension size");
      }
    }
  }
  return Status::OK();
}

int BatchingSession::RoundToLowestAllowedBatchSize(int batch_size) const {
  if (options_.allowed_batch_sizes.empty()) {
    return batch_size;
  }
  for (int allowed_size : options_.allowed_batch_sizes) {
    if (allowed_size >= batch_size) {
      return allowed_size;
    }
  }
  LOG(ERROR) << "Maximum batch size greater than largest allowed size; "
                "ignoring allowed sizes constraint";
  return batch_size;
}

Status BatchingSession::MergeInputTensors(
    const Batch<BatchingSessionTask>& batch,
    std::vector<std::pair<string, Tensor>>* merged_inputs,
    std::vector<string>* output_tensor_names) {
  DCHECK_GE(batch.num_tasks(), 1);
  if (batch.num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch.num_tasks());
  }
  *output_tensor_names = *batch.task(0).output_tensor_names;
  std::vector<string> input_tensor_names;
  for (const auto& input : *batch.task(0).inputs) {
    const string& tensor_name = input.first;
    input_tensor_names.push_back(tensor_name);
  }

  // Fast-path for a singleton batch with no padding.
  if (batch.num_tasks() == 1 && options_.allowed_batch_sizes.empty()) {
    *merged_inputs = *batch.task(0).inputs;
    return Status::OK();
  }

  const int padding_size =
      RoundToLowestAllowedBatchSize(batch.size()) - batch.size();

  for (int input_tensor_idx = 0; input_tensor_idx < input_tensor_names.size();
       ++input_tensor_idx) {
    const string& input_tensor_name = input_tensor_names[input_tensor_idx];

    std::vector<Tensor> tensors_to_merge;
    for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
      const std::vector<std::pair<string, Tensor>>& task_inputs =
          *batch.task(task_idx).inputs;
      if (task_inputs.size() != input_tensor_names.size() ||
          task_inputs[input_tensor_idx].first != input_tensor_name) {
        return errors::InvalidArgument(
            "Batching session Run() calls must supply the same input tensors");
      }
      if (input_tensor_idx == 0) {
        if (*batch.task(task_idx).output_tensor_names != *output_tensor_names) {
          return errors::InvalidArgument(
              "Batching session Run() calls must supply the same output "
              "tensors");
        }
      }
      tensors_to_merge.push_back(task_inputs[input_tensor_idx].second);
    }
    if (padding_size > 0) {
      // Use the first row of the first task's input tensor for padding.
      // (We know it exists, and represents a valid input tensor row, so it
      // should always be safe to use for padding.)
      const Tensor& first_task_tensor =
          (*batch.task(0).inputs)[input_tensor_idx].second;
      // Slice() operates on the 0th dimension, which is the batch dimension. It
      // avoids a deep copy, which is a nice efficiency bonus.
      const Tensor padding_tensor = first_task_tensor.Slice(0, 1);
      for (int i = 0; i < padding_size; ++i) {
        tensors_to_merge.push_back(padding_tensor);
      }
    }
    merged_inputs->push_back(
        {input_tensor_name, tensor::Concat(tensors_to_merge)});
  }

  return Status::OK();
}

Status BatchingSession::SplitOutputTensors(
    const std::vector<Tensor>& combined_outputs,
    Batch<BatchingSessionTask>* batch) {
  DCHECK_GE(batch->num_tasks(), 1);
  if (batch->num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch->num_tasks());
  }

  // Fast-path for a singleton batch with no padding.
  if (batch->num_tasks() == 1 && options_.allowed_batch_sizes.empty()) {
    *batch->mutable_task(0)->outputs = combined_outputs;
    return Status::OK();
  }

  std::vector<int64> task_sizes_plus_optional_padding;
  for (int i = 0; i < batch->num_tasks(); ++i) {
    task_sizes_plus_optional_padding.push_back(batch->task(i).zeroth_dim_size);
  }
  const int padding_size =
      RoundToLowestAllowedBatchSize(batch->size()) - batch->size();
  if (padding_size > 0) {
    task_sizes_plus_optional_padding.push_back(padding_size);
  }

  for (const Tensor& tensor : combined_outputs) {
    if (tensor.shape().dims() == 0) {
      return errors::FailedPrecondition(
          "Batched output tensor has 0 dimensions");
    }
    if (tensor.shape().dim_size(0) != batch->size() + padding_size) {
      return errors::FailedPrecondition(
          "Batched output tensor's 0th dimension does not equal the sum of the "
          "0th dimension sizes of the input tensors");
    }

    std::vector<Tensor> split_tensor =
        tensor::Split(tensor, task_sizes_plus_optional_padding);
    DCHECK_EQ(split_tensor.size(), task_sizes_plus_optional_padding.size());
    if (split_tensor.size() != task_sizes_plus_optional_padding.size()) {
      return errors::Internal(
          "Tensor split operation did not work as expected; got ",
          split_tensor.size(), " splits; expected ",
          task_sizes_plus_optional_padding.size());
    }

    for (int i = 0; i < batch->num_tasks(); ++i) {
      BatchingSessionTask* task = batch->mutable_task(i);
      task->outputs->push_back(split_tensor[i]);
    }
    // (Ignore a possible final split_tensor entry containing the padding.)
  }

  return Status::OK();
}

void BatchingSession::ProcessBatch(
    std::unique_ptr<Batch<BatchingSessionTask>> batch) {
  // As a possible performance optimization, consider overlapping the tensor
  // concatenation with waiting for the batch to close (i.e. do the
  // concatenation incrementally as tasks stream into the batch).
  batch->WaitUntilClosed();

  if (batch->empty()) {
    return;
  }

  Status status;

  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  auto finally = MakeCleanup([&status, &batch] {
    for (int i = 0; i < batch->num_tasks(); ++i) {
      *batch->mutable_task(i)->status = status;
      batch->mutable_task(i)->done->Notify();
    }
  });

  std::vector<std::pair<string, Tensor>> merged_inputs;
  std::vector<string> output_tensor_names;
  status = MergeInputTensors(*batch, &merged_inputs, &output_tensor_names);
  if (!status.ok()) {
    return;
  }

  std::vector<Tensor> combined_outputs;
  status = wrapped_->Run(merged_inputs, output_tensor_names,
                         {} /* target node names */, &combined_outputs);
  if (!status.ok()) {
    return;
  }

  status = SplitOutputTensors(combined_outputs, batch.get());
}

Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    std::function<
        Status(std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>,
               std::unique_ptr<BatchScheduler<BatchingSessionTask>>*)>
        batch_scheduler_creator,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session) {
  std::unique_ptr<BatchingSession> internal_batching_session;
  TF_RETURN_IF_ERROR(BatchingSession::Create(options, std::move(session),
                                             batch_scheduler_creator,
                                             &internal_batching_session));
  *batching_session = std::move(internal_batching_session);
  return Status::OK();
}

Status CreateRetryingBasicBatchingSession(
    const BasicBatchScheduler<BatchingSessionTask>::Options& schedule_options,
    const BatchSchedulerRetrier<BatchingSessionTask>::Options& retry_options,
    const BatchingSessionOptions& batching_session_options,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session) {
  if (!batching_session_options.allowed_batch_sizes.empty()) {
    if (batching_session_options.allowed_batch_sizes.back() !=
        schedule_options.max_batch_size) {
      return errors::InvalidArgument(
          "Last entry in allowed_batch_sizes must match max_batch_size; last "
          "entry was ",
          batching_session_options.allowed_batch_sizes.back(), "; expected ",
          schedule_options.max_batch_size);
    }
  }

  auto scheduler_creator = [schedule_options, retry_options](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* batch_scheduler) {
    std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>> scheduler;
    TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
        schedule_options, process_batch_callback, &scheduler));
    std::unique_ptr<BatchSchedulerRetrier<BatchingSessionTask>> retrier;
    TF_RETURN_IF_ERROR(BatchSchedulerRetrier<BatchingSessionTask>::Create(
        retry_options, std::move(scheduler), &retrier));
    *batch_scheduler = std::move(retrier);
    return Status::OK();
  };
  return CreateBatchingSession(batching_session_options, scheduler_creator,
                               std::move(session), batching_session);
}

}  // namespace serving
}  // namespace tensorflow
