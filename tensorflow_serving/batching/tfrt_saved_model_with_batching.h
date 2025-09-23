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

// Batching interface on top of TFRT SavedModel. Subject to change since TFRT
// SavedModel API is temporary and experimental.
#ifndef TENSORFLOW_SERVING_BATCHING_TFRT_SAVED_MODEL_WITH_BATCHING_H_
#define TENSORFLOW_SERVING_BATCHING_TFRT_SAVED_MODEL_WITH_BATCHING_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/batching/batching_options.h"
#include "tensorflow_serving/batching/batching_session.h"

namespace tensorflow {
namespace serving {

// The batch scheduler task type used for SavedModel batching, for use in batch
// scheduler template parameters, e.g.
// BasicBatchScheduler<SavedModelBatchingTask>.
struct SavedModelBatchingTask;

// A function to construct a batch scheduler for SavedModelBatchingTasks from a
// process-batch callback.
using SavedModelBatchingSchedulerCreator = std::function<Status(
    std::function<void(std::unique_ptr<Batch<SavedModelBatchingTask>>)>,
    std::unique_ptr<BatchScheduler<SavedModelBatchingTask>> *)>;

// A function name paired with a lambda to create a batch scheduler for Run()
// calls matching the function name.
struct FuncNameWithBatchingSchedulerCreator {
  absl::string_view func_name;
  SavedModelBatchingSchedulerCreator scheduler_creator;
};

using SavedModelBatchingOptions = BatchingOptions;

// Creates `saved_model_with_batching` that batches requests per function
// internally, where the batch scheduler for each function is created according
// to `func_name_with_batching_scheduler_creator`. `saved_model` is the
// underlying core to run inference logic and must not be null. Upon successful
// return, `saved_model_with_batching` should be used in the same way as a
// normal SavedModel. Run() call is still synchronized, and all the batching
// logic is transparent to the caller.
// Also note that the first dimension of all tensors passed to Run() must be
// batching dimension.
Status CreateSavedModelWithBatching(
    const SavedModelBatchingOptions &options,
    const std::vector<FuncNameWithBatchingSchedulerCreator>
        &func_name_with_batching_scheduler_creator,
    std::unique_ptr<tfrt::SavedModel> saved_model,
    std::unique_ptr<tfrt::SavedModel> *saved_model_with_batching);

struct SavedModelBatchingTask : public BatchingSessionTask {
  // For monitoring purpose.
  static std::string Name() { return "tfrt_saved_model_with_batching"; }

  tfrt::HostContext *host_context;
  absl::Span<const Tensor> tfrt_inputs;
  std::vector<Tensor> *tfrt_outputs;
  tfrt::SavedModel::RunOptions run_options;

  // If fields below are used, this is a partial task by splitting a large batch
  // task.
  std::vector<Tensor> tfrt_partial_inputs;

  // Status shared by all partial tasks by splitting a large batch task. The
  // original task succedds only if all partial tasks succeed.
  ThreadSafeStatus *partial_status = nullptr;
};

// The default implementation of
// `BasicBatchScheduler::Options.split_input_task_func` if corresponding batch
// scheduler for a batching session sets
// `BasicBatchScheduler::Options.enable_large_batch_splitting` to true.
Status SplitSavedModelInputTask(
    std::unique_ptr<SavedModelBatchingTask> *input_task_ptr,
    int open_batch_remaining_slot, int max_batch_size,
    std::vector<std::unique_ptr<SavedModelBatchingTask>> *output_tasks);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_TFRT_SAVED_MODEL_WITH_BATCHING_H_
