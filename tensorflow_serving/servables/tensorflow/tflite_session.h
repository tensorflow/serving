/* Copyright 2019 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_serving/batching/threadsafe_status.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/servables/tensorflow/tflite_interpreter_pool.h"

namespace tensorflow {
namespace serving {

using TensorInfoMap = std::map<string, std::pair<TensorInfo, int>>;

// Encapsulates a unit of work for BatchScheduler.
class TfLiteBatchTask : public BatchTask {
 public:
  // Creates a batch task.
  static void CreateTfLiteBatchTask(
      const std::vector<string>* output_tensor_names,
      std::vector<Tensor>* outputs, Notification* done, Status* status,
      std::unique_ptr<TfLiteBatchTask>* batch_task) {
    TfLiteBatchTask* task = new TfLiteBatchTask();
    task->is_partial = false;
    task->output_tensor_names = output_tensor_names;
    task->outputs = outputs;
    task->done = done;
    task->status = status;
    batch_task->reset(task);
  }

  // Create partial batch task.
  static void CreatePartialTfLiteBatchTask(
      std::vector<int> input_indices,
      const std::vector<string>* output_tensor_names,
      std::vector<Tensor>* outputs, std::function<void()> done_callback,
      ThreadSafeStatus* partial_status,
      std::unique_ptr<TfLiteBatchTask>* batch_task) {
    TfLiteBatchTask* task = new TfLiteBatchTask();
    task->is_partial = true;
    task->input_indices = input_indices;
    task->output_tensor_names = output_tensor_names;
    task->outputs = outputs;
    task->done_callback = done_callback;
    task->partial_status = partial_status;
    batch_task->reset(task);
  }

  TfLiteBatchTask() : enqueue_time_micros(Env::Default()->NowMicros()) {}

  TfLiteBatchTask(const TfLiteBatchTask&) = delete;

  TfLiteBatchTask& operator=(const TfLiteBatchTask&) = delete;

  ~TfLiteBatchTask() override = default;

  // Returns the batch size.
  size_t size() const override { return inputs[0].dim_size(0); }

  uint64_t start_time_micros() const { return enqueue_time_micros; }

  Notification* done;

  Status* status;

  // Input indices for the tflite tensors, aligned with inputs.
  std::vector<int> input_indices;

  // Vector of input tensors.
  std::vector<Tensor> inputs;

  // Pointer to tensor of outputs.
  std::vector<Tensor>* outputs;

  void set_output(Tensor t) { outputs->push_back(t); }

  const std::vector<string>* output_tensor_names;

  RunOptions run_options;

  const uint64_t enqueue_time_micros;

  // Required for partial execution using split batches.
  bool is_partial = false;

  // A callback for when the partial task is completed.
  std::function<void()> done_callback;

  ThreadSafeStatus* partial_status;
};

using SchedulerCreator = std::function<Status(
    const BasicBatchScheduler<TfLiteBatchTask>::Options& options,
    std::function<void(std::unique_ptr<Batch<TfLiteBatchTask>>)>,
    std::unique_ptr<BasicBatchScheduler<TfLiteBatchTask>>*)>;

// A session to run inference on a TensorFlow Lite model.
//
class TfLiteSession : public ServingSession {
 public:
  // Creates a TfLiteSession object from `buffer` representing serialized
  // TFLite flatbuffer model. Also returns the SignatureDef map based on
  // input/outputs to the model.
  //
  // run in caller thread allows a worker to run on the parent thread,
  // which may be desired to increase concurrency at the cost of additional
  // thread context overhead. Defaults to false.
  static Status Create(string&& buffer, const SessionOptions& options,
                       int num_pools, int num_interpreters_per_pool,
                       std::unique_ptr<TfLiteSession>* tflite_session,
                       ::google::protobuf::Map<string, SignatureDef>* signatures);

  static Status CreateDefaultBasicBatchScheduler(
      const BasicBatchScheduler<TfLiteBatchTask>::Options& options,
      std::function<void(std::unique_ptr<Batch<TfLiteBatchTask>>)>
          process_batch_callback,
      std::unique_ptr<BasicBatchScheduler<TfLiteBatchTask>>* batch_scheduler);

  static Status SplitTfLiteInputTask(
      std::unique_ptr<TfLiteBatchTask>* input_task_ptr,
      int open_batch_remaining_slot, int max_batch_size,
      std::vector<std::unique_ptr<TfLiteBatchTask>>* output_tasks);

  ~TfLiteSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override;

  Status SetScheduler(
      const SchedulerCreator& scheduler_creator,
      const BasicBatchScheduler<TfLiteBatchTask>::Options& options);

  BasicBatchScheduler<TfLiteBatchTask>::Options GetSchedulerOptions() {
    return scheduler_options_;
  }

 private:
  TfLiteSession(
      std::map<string, int>&& input_tensor_to_index,
      std::map<string, int>&& output_tensor_to_index, string&& buffer,
      std::unique_ptr<tflite::FlatBufferModel> model,
      std::unique_ptr<internal::TfLiteInterpreterPool> interpreter_pool);
  Status RunInternal(
      const std::vector<int>& tflite_input_indices,
      const std::vector<std::vector<const Tensor*>>& merged_inputs,
      const std::vector<string>& output_tensor_names,
      std::vector<Tensor>* combined_outputs, int batch_size,
      int* fixed_batch_size = nullptr);
  const std::map<string, int> input_tensor_to_index_;
  const std::map<string, int> output_tensor_to_index_;
  const string model_serialized_bytes_;
  const std::unique_ptr<tflite::FlatBufferModel> model_;
  const std::unique_ptr<internal::TfLiteInterpreterPool> interpreter_pool_;
  bool use_fixed_batch_size_;
  std::unique_ptr<BasicBatchScheduler<TfLiteBatchTask>> scheduler_;
  BasicBatchScheduler<TfLiteBatchTask>::Options scheduler_options_;
  void ProcessBatch(std::unique_ptr<Batch<TfLiteBatchTask>> batch);
  TF_DISALLOW_COPY_AND_ASSIGN(TfLiteSession);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_
