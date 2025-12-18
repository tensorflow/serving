/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/batching/tfrt_saved_model_with_batching.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow_serving/batching/batching_util.h"
#include "tensorflow_serving/batching/incremental_barrier.h"

namespace tensorflow {
namespace serving {

using tfrt::FunctionMetadata;
using tfrt::SavedModel;

namespace {

auto *queuing_latency = monitoring::Sampler<0>::New(
    {"/tensorflow/serving/saved_model_with_batching/queuing_latency",
     "Distribution of wall time spent (in microseconds) in queuing"},
    // Scale of 100, power of 1.2 with bucket count 52 (~1 second).
    monitoring::Buckets::Exponential(100, 1.2, 52));

// Batching implementation of SavedModel.
class SavedModelWithBatching : public tfrt::SavedModel {
 public:
  static absl::Status Create(
      const SavedModelBatchingOptions &options,
      const std::vector<FuncNameWithBatchingSchedulerCreator>
          &func_name_with_batching_scheduler_creator,
      std::unique_ptr<SavedModel> saved_model,
      std::unique_ptr<SavedModel> *result);

  // Public ctor because of absl::make_unique. It's okay given the class is
  // not publicly visible.
  SavedModelWithBatching(const SavedModelBatchingOptions &options,
                         std::unique_ptr<SavedModel> saved_model);

  const tensorflow::MetaGraphDef &GetMetaGraphDef() const override {
    return wrapped_->GetMetaGraphDef();
  }

  std::vector<std::string> GetFunctionNames() const override {
    return wrapped_->GetFunctionNames();
  }

  absl::optional<FunctionMetadata> GetFunctionMetadata(
      absl::string_view func_name) const override {
    return wrapped_->GetFunctionMetadata(func_name);
  }

  // The semantics of Run() is identical to its parent, it internally blocks,
  // batches multiple Run() and splits the result once the batch finishes and
  // unblocks.
  absl::Status Run(const tfrt::SavedModel::RunOptions &run_options,
                   absl::string_view func_name, absl::Span<const Tensor> inputs,
                   std::vector<Tensor> *outputs) override;

  absl::Status RunMultipleSignatures(
      const RunOptions &run_options, absl::Span<const std::string> names,
      absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
      std::vector<std::vector<tensorflow::Tensor>> *multi_outputs) override {
    // TODO(b/191149783): Implement batching support for
    // RunMultipleSignatures().
    return wrapped_->RunMultipleSignatures(run_options, names, multi_inputs,
                                           multi_outputs);
  }

  absl::Status RunByTensorNames(
      const RunOptions &run_options,
      absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
      absl::Span<const std::string> output_tensor_names,
      absl::Span<const std::string> target_node_names,
      std::vector<tensorflow::Tensor> *outputs) {
    // TODO(b/191149783): Implement batching support for RunByTensorNames().
    return wrapped_->RunByTensorNames(run_options, inputs, output_tensor_names,
                                      target_node_names, outputs);
  }

 private:
  // Batches tensors in `batch` and invokes Run() with underlying `wrapped_`.
  void ProcessBatch(absl::string_view func_name,
                    std::unique_ptr<Batch<SavedModelBatchingTask>> batch);

  // Batches tensors in `batch` and stores the result in `batch_inputs`.
  absl::Status BatchInputTensors(absl::string_view func_name,
                                 const Batch<SavedModelBatchingTask> &batch,
                                 std::vector<Tensor> *batch_inputs);

  // For each tensor in `combined_outputs`, splits it according to `batch` and
  // stores the result in corresponding BatchingTask.
  absl::Status SplitOutputTensors(std::vector<Tensor> combined_outputs,
                                  Batch<SavedModelBatchingTask> *batch);

  const SavedModelBatchingOptions options_;

  // Underlying SavedModel.
  std::unique_ptr<SavedModel> wrapped_;
  absl::flat_hash_map<std::string,
                      std::unique_ptr<BatchScheduler<SavedModelBatchingTask>>>
      batch_schedulers_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelWithBatching);
};

SavedModelWithBatching::SavedModelWithBatching(
    const SavedModelBatchingOptions &options,
    std::unique_ptr<SavedModel> saved_model)
    : tfrt::SavedModel(&saved_model->runtime()),
      options_(options),
      wrapped_(std::move(saved_model)) {}

absl::Status SavedModelWithBatching::Create(
    const SavedModelBatchingOptions &options,
    const std::vector<FuncNameWithBatchingSchedulerCreator>
        &func_name_with_batching_scheduler_creators,
    std::unique_ptr<SavedModel> saved_model,
    std::unique_ptr<SavedModel> *result) {
  if (saved_model == nullptr) {
    return errors::FailedPrecondition("saved_model must not be null.");
  }

  SavedModel *raw_saved_model = saved_model.get();
  std::unique_ptr<SavedModelWithBatching> saved_model_with_batching =
      absl::make_unique<SavedModelWithBatching>(options,
                                                std::move(saved_model));
  SavedModelWithBatching *raw_saved_model_with_batching =
      saved_model_with_batching.get();

  for (const auto &entry : func_name_with_batching_scheduler_creators) {
    if (!raw_saved_model->GetFunctionMetadata(entry.func_name)) {
      LOG(WARNING) << "Function " << entry.func_name
                   << " is not found in the model. ";
      continue;
    }

    auto insert_result = saved_model_with_batching->batch_schedulers_.emplace(
        std::string(entry.func_name), /*scheduler=*/nullptr);
    if (!insert_result.second) {
      return errors::FailedPrecondition(
          absl::StrCat("Specified multiple batch schedulers for function ",
                       entry.func_name));
    }

    const std::string &func_name = insert_result.first->first;
    TF_RETURN_IF_ERROR(entry.scheduler_creator(
        [func_name, raw_saved_model_with_batching](
            std::unique_ptr<Batch<SavedModelBatchingTask>> batch) {
          raw_saved_model_with_batching->ProcessBatch(func_name,
                                                      std::move(batch));
        },
        &insert_result.first->second));
    if (insert_result.first->second == nullptr) {
      return errors::FailedPrecondition(absl::StrCat(
          "Failed to create batch scheduler for function ", entry.func_name));
    }
  }
  *result = std::move(saved_model_with_batching);
  return absl::Status();
}

absl::Status SavedModelWithBatching::Run(
    const tfrt::SavedModel::RunOptions &run_options,
    absl::string_view func_name, absl::Span<const Tensor> inputs,
    std::vector<Tensor> *outputs) {
  if (outputs == nullptr) {
    return errors::FailedPrecondition("outputs must not be null");
  }
  auto it = batch_schedulers_.find(func_name);
  if (it == batch_schedulers_.end()) {
    // Batch scheduler not found for this function, run it with underlying
    // SavedModel in-line.
    static uint64_t last_log_message_secs = 0;
    // Not thread safe, but that's how batching_session is doing as well.
    // TODO(b/168220822): It probably matters, what if last_log_message_secs
    // mistakenly becomes too large?
    uint64_t now_secs = EnvTime::NowSeconds();
    if (now_secs - last_log_message_secs >= 120) {
      LOG(WARNING) << "Request doesn't match any declared function. Bypassing "
                      "batcher. Request function is: "
                   << func_name;
      last_log_message_secs = now_secs;
    }
    return wrapped_->Run(run_options, func_name, inputs, outputs);
  }
  outputs->clear();

  absl::Notification done;
  absl::Status status;
  auto task = absl::make_unique<SavedModelBatchingTask>();
  TF_RETURN_IF_ERROR(ComputeTensorBatchSize(
      inputs, &task->zeroth_dim_size,
      [](const Tensor &tensor) { return tensor.dims(); },
      [](const Tensor &tensor, size_t dim) { return tensor.dim_size(dim); }));
  RecordInputBatchSize<SavedModelBatchingTask>(task->zeroth_dim_size);

  task->host_context = GetHostContext();
  task->tfrt_inputs = inputs;
  task->tfrt_outputs = outputs;
  task->done = &done;
  task->status = &status;
  task->run_options = run_options;
  task->enqueue_time_micros = EnvTime::NowMicros();

  TF_RETURN_IF_ERROR(it->second->Schedule(&task));
  done.WaitForNotification();
  return status;
}

// TODO(b/168220822): Once tfrt supports tensor split/pad/concat utilities and
// removes llvm dependency, refactors this function accordingly (return type may
// change).
std::vector<absl::InlinedVector<int, 4>> CalculateMaxDimSizes(
    const Batch<SavedModelBatchingTask> &batch) {
  std::vector<absl::InlinedVector<int, 4>> max_dim_sizes;
  for (int batch_idx = 0; batch_idx < batch.num_tasks(); ++batch_idx) {
    const auto inputs = batch.task(batch_idx).tfrt_inputs;
    for (int tensor_idx = 0; tensor_idx < inputs.size(); ++tensor_idx) {
      const Tensor &tensor = inputs[tensor_idx];
      const TensorShape &shape = tensor.shape();
      const int rank = shape.dims();

      absl::InlinedVector<int, 4> dims;
      dims.reserve(rank);
      for (auto dim : shape) {
        dims.push_back(dim.size);
      }

      if (batch_idx == 0) {
        max_dim_sizes.push_back(std::move(dims));
      } else {
        for (int rank_idx = 0; rank_idx < rank; ++rank_idx) {
          int &cur_max_size = max_dim_sizes[tensor_idx][rank_idx];
          cur_max_size = std::max(cur_max_size, dims[rank_idx]);
        }
      }
    }
  }
  return max_dim_sizes;
}

absl::Status SavedModelWithBatching::BatchInputTensors(
    absl::string_view func_name, const Batch<SavedModelBatchingTask> &batch,
    std::vector<Tensor> *batch_inputs) {
  if (batch.num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch.num_tasks());
  }
  const int original_batch_size = batch.size();
  const int target_batch_size = RoundToLowestAllowedBatchSize(
      options_.allowed_batch_sizes, original_batch_size);
  const int padding_size = target_batch_size - original_batch_size;
  RecordPaddingSize<SavedModelBatchingTask>(padding_size, target_batch_size);
  RecordProcessedBatchSize<SavedModelBatchingTask>(target_batch_size);

  std::vector<absl::InlinedVector<int, 4>> max_dim_sizes;
  if (options_.pad_variable_length_inputs) {
    max_dim_sizes = CalculateMaxDimSizes(batch);
  }

  // TODO(b/168220822): Padding logic below operates on tfrt inputs. It's pretty
  // much a duplicate of batching session Padding logic. Rewrite once TFRT
  // tensor supports necessary utilities.
  std::vector<std::vector<Tensor>> tensors_to_merge(
      batch.task(0).tfrt_inputs.size(), std::vector<Tensor>());
  for (int batch_idx = 0; batch_idx < batch.num_tasks(); ++batch_idx) {
    auto inputs = batch.task(batch_idx).tfrt_inputs;

    for (int tensor_idx = 0; tensor_idx < inputs.size(); ++tensor_idx) {
      Tensor tensor = inputs[tensor_idx];
      std::vector<Tensor> &tensor_vec = tensors_to_merge[tensor_idx];

      Tensor optionally_padded_tensor;
      if (options_.pad_variable_length_inputs) {
        TF_RETURN_IF_ERROR(AddPadding(tensor, max_dim_sizes[tensor_idx],
                                      &optionally_padded_tensor));
      } else {
        optionally_padded_tensor = tensor;
        if (batch_idx > 0) {
          TensorShape reference_shape = tensors_to_merge[tensor_idx][0].shape();

          if (!AreShapesEqualExceptZeroDim(tensor.shape(), reference_shape)) {
            return errors::FailedPrecondition(
                " Tensors in a single batch have different shapes other than"
                " first dimension and padding is turned off.");
          }
        }
      }
      tensor_vec.push_back(std::move(optionally_padded_tensor));

      if (batch_idx == batch.num_tasks() - 1 && padding_size > 0) {
        const Tensor padding_tensor = tensor_vec.back().Slice(0, 1);
        for (int i = 0; i < padding_size; ++i) {
          tensor_vec.push_back(padding_tensor);
        }
      }
    }
  }

  for (const auto &tensors : tensors_to_merge) {
    Tensor concated;
    TF_RETURN_IF_ERROR(tensor::Concat(tensors, &concated));
    batch_inputs->push_back(concated);
  }

  return absl::Status();
}

void SavedModelWithBatching::ProcessBatch(
    absl::string_view func_name,
    std::unique_ptr<Batch<SavedModelBatchingTask>> batch) {
  batch->WaitUntilClosed();

  if (batch->empty()) return;
  absl::Status status = absl::Status();
  auto cleanup = gtl::MakeCleanup([&status, &batch] {
    for (int batch_idx = 0; batch_idx < batch->num_tasks(); ++batch_idx) {
      SavedModelBatchingTask *task = batch->mutable_task(batch_idx);
      if (task->partial_status != nullptr) {
        task->partial_status->Update(status);
        task->done_callback();
      } else {
        *(task->status) = status;
        task->done->Notify();
      }
    }
  });

  const uint64_t dequeue_time_micros = EnvTime::NowMicros();

  bool all_tasks_timeout_exceeded = true;
  absl::optional<std::chrono::system_clock::time_point> batch_deadline;
  for (int batch_idx = 0; batch_idx < batch->num_tasks(); ++batch_idx) {
    const SavedModelBatchingTask &task = batch->task(batch_idx);
    if (!task.run_options.deadline.has_value() ||
        absl::ToChronoTime(absl::Now()) < task.run_options.deadline.value()) {
      all_tasks_timeout_exceeded = false;
      if (task.run_options.deadline.has_value() &&
          (!batch_deadline.has_value() ||
           batch_deadline.value() < task.run_options.deadline.value())) {
        batch_deadline = task.run_options.deadline;
      }
    }
    queuing_latency->GetCell()->Add(dequeue_time_micros -
                                    task.enqueue_time_micros);
  }

  if (all_tasks_timeout_exceeded) {
    status = absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kResourceExhausted),
        "Run() timeout exceeded while waiting in batching queue");
    return;
  }

  tfrt::SavedModel::RunOptions batch_run_options;
  batch_run_options.deadline = batch_deadline;
  std::vector<Tensor> batch_inputs;
  status = BatchInputTensors(func_name, *batch, &batch_inputs);
  if (!status.ok()) return;

  std::vector<Tensor> combined_outputs;
  status = wrapped_->Run(batch_run_options, func_name, batch_inputs,
                         &combined_outputs);
  if (!status.ok()) return;
  status = SplitOutputTensors(std::move(combined_outputs), batch.get());
}

absl::Status SavedModelWithBatching::SplitOutputTensors(
    std::vector<Tensor> combined_outputs,
    Batch<SavedModelBatchingTask> *batch) {
  std::vector<int64_t> split_batch_sizes;
  split_batch_sizes.reserve(batch->num_tasks());
  for (int batch_idx = 0; batch_idx < batch->num_tasks(); ++batch_idx) {
    split_batch_sizes.push_back(batch->task(batch_idx).size());
  }
  const int64_t no_padded_batch_size = batch->size();
  const int64_t padded_batch_size = RoundToLowestAllowedBatchSize(
      options_.allowed_batch_sizes, no_padded_batch_size);

  const int64_t padding_size = padded_batch_size - no_padded_batch_size;
  if (padding_size > 0) {
    split_batch_sizes.push_back(padding_size);
  }

  for (const auto &combined_tensor : combined_outputs) {
    std::vector<Tensor> split_tensors;
    TF_RETURN_IF_ERROR(
        tensor::Split(combined_tensor, split_batch_sizes, &split_tensors));

    for (int batch_idx = 0; batch_idx < batch->num_tasks(); ++batch_idx) {
      SavedModelBatchingTask *task = batch->mutable_task(batch_idx);
      task->tfrt_outputs->push_back(split_tensors.at(batch_idx));
    }
  }
  return absl::Status();
}

}  // namespace

absl::Status CreateSavedModelWithBatching(
    const SavedModelBatchingOptions &options,
    const std::vector<FuncNameWithBatchingSchedulerCreator>
        &func_name_with_batching_scheduler_creator,
    std::unique_ptr<tfrt::SavedModel> saved_model,
    std::unique_ptr<tfrt::SavedModel> *saved_model_with_batching) {
  return SavedModelWithBatching::Create(
      options, func_name_with_batching_scheduler_creator,
      std::move(saved_model), saved_model_with_batching);
}

absl::Status SplitSavedModelInputTask(
    std::unique_ptr<SavedModelBatchingTask> *input_task_ptr,
    int open_batch_remaining_slot, int max_batch_size,
    std::vector<std::unique_ptr<SavedModelBatchingTask>> *output_tasks) {
  SavedModelBatchingTask *input_task = input_task_ptr->get();

  // TODO(b/168220822): Also split RunMetadata once TFRT supports it.

  // Each inner vector will be passed to a partial task thus needs to be
  // unique_ptr. shared_ptr because std::function is not compatible with
  // capture by move.
  auto split_output =
      std::make_shared<std::vector<std::unique_ptr<std::vector<Tensor>>>>();
  auto partial_status = std::make_shared<ThreadSafeStatus>();

  auto split_task_done_callback = [split_output, partial_status,
                                   status = input_task->status,
                                   output = input_task->tfrt_outputs,
                                   done_notification = input_task->done]() {
    auto cleanup = gtl::MakeCleanup(
        [done_notification]() { done_notification->Notify(); });

    // Fail early if any partial task fails.
    if (!partial_status->status().ok()) {
      *status = partial_status->status();
      return;
    }

    int output_size = split_output->size();
    int tensor_size = (*split_output)[0]->size();
    for (int tensor_idx = 0; tensor_idx < tensor_size; ++tensor_idx) {
      Tensor output_tensor;
      std::vector<Tensor> to_concatenate;
      to_concatenate.reserve(output_size);
      for (int output_idx = 0; output_idx < output_size; ++output_idx) {
        to_concatenate.push_back(
            std::move((*(*split_output)[output_idx])[tensor_idx]));
      }
      const auto concat_status = tensor::Concat(to_concatenate, &output_tensor);
      if (!concat_status.ok()) {
        *status = concat_status;
        return;
      }
      output->push_back(output_tensor);
    }
    *status = absl::Status();
  };

  // The Callback will be run only after all partial tasks finished.
  IncrementalBarrier barrier(std::move(split_task_done_callback));
  std::vector<int64_t> output_task_sizes;

  if (open_batch_remaining_slot > 0) {
    output_task_sizes.push_back(open_batch_remaining_slot);
    split_output->emplace_back(absl::make_unique<std::vector<Tensor>>());
  }

  for (int left_task_size = input_task->size() - open_batch_remaining_slot;
       left_task_size > 0; left_task_size -= max_batch_size) {
    int next_task_size = std::min(left_task_size, max_batch_size);
    output_task_sizes.push_back(next_task_size);
    split_output->emplace_back(absl::make_unique<std::vector<Tensor>>());
  }

  const int output_task_num = output_task_sizes.size();

  // Construct partial tasks.
  output_tasks->reserve(output_task_num);
  for (int i = 0; i < output_task_num; ++i) {
    auto task = absl::make_unique<SavedModelBatchingTask>();
    task->zeroth_dim_size = output_task_sizes[i];
    task->run_options = input_task->run_options;
    task->tfrt_outputs = (*split_output)[i].get();
    task->done_callback = barrier.Inc();
    task->partial_status = partial_status.get();
    output_tasks->push_back(std::move(task));
  }

  for (const Tensor &input : input_task->tfrt_inputs) {
    std::vector<Tensor> split_tensors;
    TF_RETURN_IF_ERROR(tensor::Split(input, output_task_sizes, &split_tensors));
    for (int output_idx = 0; output_idx < output_task_num; ++output_idx) {
      auto &output_task = (*output_tasks)[output_idx];
      output_task->tfrt_partial_inputs.push_back(split_tensors[output_idx]);
    }
  }

  for (auto &task : *output_tasks) {
    task->tfrt_inputs = task->tfrt_partial_inputs;
  }

  return absl::Status();
}

}  // namespace serving
}  // namespace tensorflow
