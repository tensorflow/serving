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

#include <memory>

#include "absl/container/fixed_array.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/batching/batching_util.h"
#include "tensorflow_serving/batching/incremental_barrier.h"
#include "tensorflow_serving/batching/threadsafe_status.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/util/hash.h"

namespace tensorflow {
namespace serving {

namespace {

auto* queuing_latency = monitoring::Sampler<1>::New(
    {"/tensorflow/serving/batching_session/queuing_latency",
     "Distribution of wall time spent (in microseconds) in queuing",
     "thread_pool_name"},
    // Scale of 100, power of 1.2 with bucket count 52 (~1 second).
    monitoring::Buckets::Exponential(100, 1.2, 52));

auto* wrapped_run_count = monitoring::Counter<0>::New(
    "/tensorflow/serving/batching_session/wrapped_run_count",
    "Total count of run calls on the wrapped session");

string TensorSignatureDebugString(const TensorSignature& signature) {
  return strings::StrCat("{input_tensors: <",
                         str_util::Join(signature.input_tensors, ", "),
                         ">, output_tensors: <",
                         str_util::Join(signature.output_tensors, ", "), ">}");
}

struct HashTensorSignature {
  uint64_t operator()(const TensorSignature& signature) const {
    uint64_t hash = 0xDECAFCAFFE /* seed */;
    for (const string& input_tensor : signature.input_tensors) {
      hash = HashCombine(hash, std::hash<string>()(input_tensor));
    }
    for (const string& output_tensor : signature.output_tensors) {
      hash = HashCombine(hash, std::hash<string>()(output_tensor));
    }
    return hash;
  }
};

struct EqTensorSignature {
  bool operator()(const TensorSignature& lhs,
                  const TensorSignature& rhs) const {
    return lhs.input_tensors == rhs.input_tensors &&
           lhs.output_tensors == rhs.output_tensors;
  }
};

// Constructs a TensorSignature from a Run() call's 'inputs' and
// 'output_tensor_names' arguments.
TensorSignature TensorSignatureFromRunArgs(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names) {
  TensorSignature signature;
  for (const auto& entry : inputs) {
    const string& tensor_name = entry.first;
    signature.input_tensors.insert(tensor_name);
  }
  for (const string& output_tensor_name : output_tensor_names) {
    signature.output_tensors.insert(output_tensor_name);
  }
  return signature;
}

// Constructs vector of one task input from one BatchingSessionTask.
const std::vector<std::pair<string, Tensor>>& GetTaskInput(
    const BatchingSessionTask& batching_session_task) {
  if (batching_session_task.is_partial) {
    return *batching_session_task.owned_split_inputs;
  }
  return *batching_session_task.inputs;
}

// Constructs vector of all task inputs from Batch of BatchingSessionTasks.
// Input for each task is a vector of pairs (tensor_name, tensor_value).
std::vector<std::vector<std::pair<string, Tensor>>> GetTaskInputsVector(
    const Batch<BatchingSessionTask>& batch) {
  std::vector<std::vector<std::pair<string, Tensor>>> all_task_inputs;
  all_task_inputs.reserve(batch.num_tasks());
  for (int i = 0; i < batch.num_tasks(); ++i) {
    all_task_inputs.push_back(GetTaskInput(batch.task(i)));
  }
  return all_task_inputs;
}

}  // namespace

TensorSignature TensorSignatureFromSignatureDef(
    const SignatureDef& signature_def) {
  return TensorSignatureFromSignatureDefs({signature_def});
}

TensorSignature TensorSignatureFromSignatureDefs(
    const std::vector<SignatureDef>& signature_defs) {
  TensorSignature tensor_signature;
  for (const SignatureDef& signature_def : signature_defs) {
    for (const auto& entry : signature_def.inputs()) {
      const TensorInfo& tensor_info = entry.second;
      tensor_signature.input_tensors.insert(tensor_info.name());
    }
    for (const auto& entry : signature_def.outputs()) {
      const TensorInfo& tensor_info = entry.second;
      tensor_signature.output_tensors.insert(tensor_info.name());
    }
  }
  return tensor_signature;
}

// A session that performs batching on top of a wrapped session. See the
// documentation in batching_session.h for details and constraints.
class BatchingSession : public ServingSession {
 public:
  // Constructs a BatchingSession. Arguments:
  // - 'options' contains parameters. See batching_session.h.
  // - 'wrapped' is the session to wrap with batching.
  // - 'signatures_with_scheduler_creators' specifies the set of supported
  //   signatures, and for each one supplies a lambda to construct a batch
  //   scheduler given a process-batch callback. See batching_session.h for
  //   example usage.
  static Status Create(
      const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
      const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
          signatures_with_scheduler_creators,
      const std::string& thread_pool_name,
      std::unique_ptr<BatchingSession>* result);

  // Same as above but allows for specification of a default scheduler creator
  // which enables requests that don't match an exact signature to also
  // have batching.
  static Status Create(
      const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
      const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
          signatures_with_scheduler_creators,
      BatchingSessionSchedulerCreator default_creator,
      const std::string& thread_pool_name,
      std::unique_ptr<BatchingSession>* result);

  ~BatchingSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  // RunOptions handling:
  // Since multiple of these Run() calls get backed into a single call to the
  // underlying Session's Run(), we select an arbitrary 'run_options' (typically
  // they are the same across calls). The exception is the timeout; we take the
  // largest value (after subtracting time spent in the batching queue).
  //
  // RunMetadata:
  // We copy the batched call's RunMetadata to each non-batched call's output.
  // When input of a call is processed in multiple batches as opposed to one
  // (i.e., `enable_large_batch_splitting` is true for batch scheduler),
  // `RunMetadata.CostGraphDef.AggregatedCost` is the sum of all splits of the
  // corresponding input and as correct as if the input is not split (again
  // assuming all individual tasks in a batch have equal cost, which is the
  // assumption before splitting is introduced), the rest of fields in
  // `RunMetadata` are copied from the processing result of first split.
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  // Similar to the function above, but takes an additional
  // 'thread_pool_options' to pass to the underlying Session's Run(). We select
  // an arbitrary 'thread_pool_options' (typically they are the same across
  // calls).
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override;

 private:
  explicit BatchingSession(const BatchingSessionOptions& options,
                           const std::string& thread_pool_name);

  // Helper fucntion to run the session.
  Status InternalRun(
      const RunOptions& run_options,
      const std::vector<std::pair<string, Tensor>>& inputs,
      const std::vector<string>& output_tensor_names,
      const std::vector<string>& target_node_names,
      std::vector<Tensor>* outputs, RunMetadata* run_metadata,
      absl::optional<thread::ThreadPoolOptions> thread_pool_options);

  // Computes the size of an input tensor list for batching purposes, by
  // analyzing the 0th dimension size of each of the tensors. All tensors in the
  // list must have the same 0th dimension size to be batchable. If the sizes
  // are not all identical, returns an error.
  Status ComputeInputSize(const std::vector<std::pair<string, Tensor>>& inputs,
                          size_t* size) const;

  // Merges the input tensors in a batch, via concatenation of correspondingly-
  // named tensors. Puts the merged inputs in the order they are in in the
  // signature. Assumes 'batch' is non-empty. Returns an error if there are any
  // mismatches among the tasks in the batch that violate the constraints for
  // batchability.
  Status MergeInputTensors(
      const TensorSignature& signature, const Batch<BatchingSessionTask>& batch,
      std::vector<std::pair<string, Tensor>>* merged_inputs);

  // Splits the output of a batched call to 'wrapped_->Run()' into individual
  // task outputs. Assumes the output tensor order matches the signature.
  Status SplitOutputTensors(const TensorSignature& signature,
                            const std::vector<Tensor>& combined_outputs,
                            Batch<BatchingSessionTask>* batch);

  // Splits RunMetadata parts (e.g. costgraph attribution) into individual task
  // outputs.
  Status SplitRunMetadata(RunMetadata* batch_metadata,
                          Batch<BatchingSessionTask>* batch);

  // Processes one batch of Run() calls with 'signature'. Called by
  // 'batch_scheduler_' in a batch thread.
  void ProcessBatch(const TensorSignature& signature,
                    std::unique_ptr<Batch<BatchingSessionTask>> batch);

  const BatchingSessionOptions options_;
  // The name of the thread pool of the underlying batch scheduler. It is used
  // for monitoring purpose, and can be empty if not known.
  const std::string thread_pool_name_;

  std::unique_ptr<Session> wrapped_;
  std::unordered_map<TensorSignature,
                     std::unique_ptr<BatchScheduler<BatchingSessionTask>>,
                     HashTensorSignature, EqTensorSignature>
      batch_schedulers_;

  // If set, default_scheduler_creator_ is used when the input signature does
  // not match any existing signature defined during model load. This helps
  // when the user uses either a combination of signatures or filter certain
  // output tensors.
  absl::optional<BatchingSessionSchedulerCreator> default_scheduler_creator_;
  absl::Mutex mu_;
  std::unordered_map<TensorSignature,
                     std::unique_ptr<BatchScheduler<BatchingSessionTask>>,
                     HashTensorSignature, EqTensorSignature>
      custom_signature_batch_schedulers_ ABSL_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(BatchingSession);
};

Status BatchingSession::Create(
    const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    BatchingSessionSchedulerCreator default_creator,
    const std::string& thread_pool_name,
    std::unique_ptr<BatchingSession>* result) {
  auto status = BatchingSession::Create(options, std::move(wrapped),
                                        signatures_with_scheduler_creators,
                                        thread_pool_name, result);
  result->get()->default_scheduler_creator_ = default_creator;
  return status;
}

Status BatchingSession::Create(
    const BatchingSessionOptions& options, std::unique_ptr<Session> wrapped,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    const std::string& thread_pool_name,
    std::unique_ptr<BatchingSession>* result) {
  auto batching_session = std::unique_ptr<BatchingSession>(
      new BatchingSession(options, thread_pool_name));
  BatchingSession* raw_batching_session = batching_session.get();
  batching_session->wrapped_ = std::move(wrapped);

  for (const auto& entry : signatures_with_scheduler_creators) {
    const TensorSignature& signature = entry.signature;
    const BatchingSessionSchedulerCreator& scheduler_creator =
        entry.scheduler_creator;

    std::unique_ptr<BatchScheduler<BatchingSessionTask>> batch_scheduler;
    TF_RETURN_IF_ERROR(scheduler_creator(
        [signature, raw_batching_session](
            std::unique_ptr<Batch<BatchingSessionTask>> batch) {
          raw_batching_session->ProcessBatch(signature, std::move(batch));
        },
        &batch_scheduler));
    batching_session->batch_schedulers_[signature] = std::move(batch_scheduler);
  }

  *result = std::move(batching_session);
  return OkStatus();
}

Status BatchingSession::Run(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_tensor_names, target_node_names,
             outputs, &run_metadata);
}

Status BatchingSession::Run(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata) {
  return InternalRun(run_options, inputs, output_tensor_names,
                     target_node_names, outputs, run_metadata, absl::nullopt);
}

Status BatchingSession::Run(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& thread_pool_options) {
  return InternalRun(run_options, inputs, output_tensor_names,
                     target_node_names, outputs, run_metadata,
                     thread_pool_options);
}

Status BatchingSession::InternalRun(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    absl::optional<thread::ThreadPoolOptions> thread_pool_options) {
  if (!target_node_names.empty()) {
    return errors::PermissionDenied(
        "BatchingSession does not support target nodes");
  }

  profiler::TraceMe trace_me([this] {
    return profiler::TraceMeEncode(
        "BatchingSessionRun",
        {{"thread_pool_name", thread_pool_name_}, {"_r", 1} /*root_event*/});
  });
  const TensorSignature signature =
      TensorSignatureFromRunArgs(inputs, output_tensor_names);
  auto batch_scheduler_it = batch_schedulers_.find(signature);
  if (batch_scheduler_it == batch_schedulers_.end()) {
    if (default_scheduler_creator_.has_value()) {
      absl::MutexLock l(&mu_);
      batch_scheduler_it = custom_signature_batch_schedulers_.find(signature);
      if (batch_scheduler_it == custom_signature_batch_schedulers_.end()) {
        std::unique_ptr<BatchScheduler<BatchingSessionTask>> batch_scheduler;
        TF_RETURN_IF_ERROR(default_scheduler_creator_.value()(
            [&, signature](std::unique_ptr<Batch<BatchingSessionTask>> batch) {
              ProcessBatch(signature, std::move(batch));
            },
            &batch_scheduler));
        custom_signature_batch_schedulers_[signature] =
            std::move(batch_scheduler);
        batch_scheduler_it = custom_signature_batch_schedulers_.find(signature);
      }
    } else {
      // We have a Run() call that doesn't match one of our batching signatures.
      // Run it in-line.
      LOG_EVERY_N_SEC(WARNING, 120)
          << "Request doesn't match any declared signature and no default "
             "scheduler creator specified. Bypassing "
             "batcher. Request signature is: "
          << TensorSignatureDebugString(signature);

      // Because the wrapped session may not provide an implementation for
      // thread_pool_options, we need to invoke different Run() functions
      // depending on whether thread_pool_options is specified.
      if (thread_pool_options) {
        return wrapped_->Run(run_options, inputs, output_tensor_names,
                             target_node_names, outputs, run_metadata,
                             thread_pool_options.value());
      } else {
        return wrapped_->Run(run_options, inputs, output_tensor_names,
                             target_node_names, outputs, run_metadata);
      }
    }
  }
  BatchScheduler<BatchingSessionTask>* batch_scheduler =
      batch_scheduler_it->second.get();

  outputs->clear();

  Notification done;
  Status status;
  auto task = std::unique_ptr<BatchingSessionTask>(new BatchingSessionTask);
  task->enqueue_time_micros = EnvTime::NowMicros();
  task->run_options = run_options;
  TF_RETURN_IF_ERROR(ComputeInputSize(inputs, &task->zeroth_dim_size));
  task->inputs = &inputs;
  task->output_tensor_names = &output_tensor_names;
  task->done = &done;
  task->status = &status;
  task->outputs = outputs;
  task->run_metadata = run_metadata;
  task->thread_pool_options = thread_pool_options;
  task->thread_safe_status = std::make_shared<ThreadSafeStatus>();
  task->shared_outputs = std::make_shared<std::vector<std::vector<Tensor>>>();
  task->split_run_metadatas = absl::make_unique<std::vector<RunMetadata>>();

  TF_RETURN_IF_ERROR(batch_scheduler->Schedule(&task));
  done.WaitForNotification();
  return status;
}

Status BatchingSession::ListDevices(std::vector<DeviceAttributes>* response) {
  return wrapped_->ListDevices(response);
}

BatchingSession::BatchingSession(const BatchingSessionOptions& options,
                                 const std::string& thread_pool_name)
    : options_(options), thread_pool_name_(thread_pool_name) {}

Status BatchingSession::ComputeInputSize(
    const std::vector<std::pair<string, Tensor>>& inputs, size_t* size) const {
  TF_RETURN_IF_ERROR(::tensorflow::serving::ComputeTensorBatchSize(
      inputs, size,
      [](const std::pair<std::string, Tensor>& tensor) {
        return tensor.second.shape().dims();
      },
      [](const std::pair<std::string, Tensor>& tensor, size_t dim) {
        return tensor.second.shape().dim_size(dim);
      }));
  for (const auto& entry : inputs) {
    const Tensor& tensor = entry.second;
    RecordInputBatchSize<BatchingSessionTask>(tensor.shape().dim_size(0));
  }
  return OkStatus();
}

Status BatchingSession::MergeInputTensors(
    const TensorSignature& signature, const Batch<BatchingSessionTask>& batch,
    std::vector<std::pair<string, Tensor>>* merged_inputs) {
  DCHECK_GE(batch.num_tasks(), 1);
  if (batch.num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch.num_tasks());
  }

  const int lowest_allowed_batch_size =
      RoundToLowestAllowedBatchSize(options_.allowed_batch_sizes, batch.size());
  const int padding_size = lowest_allowed_batch_size - batch.size();
  profiler::TraceMe trace_me([lowest_allowed_batch_size, padding_size]() {
    return profiler::TraceMeEncode(
        "MergeInputTensors",
        {{"batch_size_after_padding", lowest_allowed_batch_size},
         {"padding_amount", padding_size}});
  });
  RecordPaddingSize<BatchingSessionTask>(padding_size,
                                         lowest_allowed_batch_size);
  RecordProcessedBatchSize<BatchingSessionTask>(lowest_allowed_batch_size);

  // For each input tensor name, a vector of tensors from the individual tasks.
  std::map<string, std::vector<Tensor>> tensors_to_merge;
  // For each input tensor name a vector of maximum dimension sizes
  // among tensors from individual tasks.
  absl::optional<std::map<string, std::vector<int>>> max_dim_sizes;
  if (options_.pad_variable_length_inputs) {
    std::vector<std::vector<std::pair<string, Tensor>>> all_task_inputs =
        GetTaskInputsVector(batch);
    max_dim_sizes = CalculateMaxDimSizes(all_task_inputs);
  }
  // Populate 'tensors_to_merge'.
  for (int i = 0; i < batch.num_tasks(); ++i) {
    const std::vector<std::pair<string, Tensor>>& task_inputs =
        GetTaskInput(batch.task(i));
    for (const auto& entry : task_inputs) {
      const string& tensor_name = entry.first;
      const Tensor& tensor = entry.second;

      std::vector<Tensor>& tensor_vec = tensors_to_merge[tensor_name];
      Tensor optionally_padded_tensor;
      if (options_.pad_variable_length_inputs) {
        TF_RETURN_IF_ERROR(AddPadding(tensor, (*max_dim_sizes)[tensor_name],
                                      &optionally_padded_tensor));
      } else {
        optionally_padded_tensor = tensor;
        // Check whether tensors with the same name have equal dims
        // (except zeroth dim) when padding is turned off.
        if (i > 0) {  // added at least one task to tensors_to_merge
          TensorShape reference_shape =
              tensors_to_merge[tensor_name][0].shape();
          if (!AreShapesEqualExceptZeroDim(tensor.shape(), reference_shape)) {
            return errors::FailedPrecondition(
                "Tensors with name '" + tensor_name +
                "' from different tasks have different shapes and padding is "
                "turned off. Set pad_variable_length_inputs to true, or ensure "
                "that all tensors with the same name have equal dimensions "
                "starting with the first dim.");
          }
        }
      }
      tensor_vec.push_back(std::move(optionally_padded_tensor));
      if (i == batch.num_tasks() - 1 && padding_size > 0) {
        // This is the last task. Insert padding.
        //
        // Use the first row of this task's tensor as the padding data. (We know
        // it represents a valid input tensor row, so it should always be safe
        // to use for padding.)
        //
        // Slice() operates on the 0th dimension, which is the batch dimension.
        // It avoids a deep copy, which is a nice efficiency bonus.
        const Tensor padding_tensor = tensor_vec.back().Slice(0, 1);
        for (int i = 0; i < padding_size; ++i) {
          tensor_vec.push_back(padding_tensor);
        }
      }
    }
  }

  // Merge the tensors.
  DCHECK_EQ(signature.input_tensors.size(), tensors_to_merge.size());
  if (tensors_to_merge.size() != signature.input_tensors.size()) {
    return errors::Internal(
        "One or more tasks does not conform to batch signature");
  }
  for (const string& tensor_name : signature.input_tensors) {
    auto tensors = tensors_to_merge.find(tensor_name);
    DCHECK(tensors != tensors_to_merge.end());
    if (tensors == tensors_to_merge.end()) {
      return errors::Internal(
          "One or more tasks does not conform to batch signature");
    }
    Tensor concated;
    const Status concat_status = tensor::Concat(tensors->second, &concated);
    DCHECK(concat_status.ok()) << concat_status.ToString();
    if (!concat_status.ok()) {
      return errors::Internal("Tensor concat operation failed: ",
                              concat_status.ToString());
    }
    merged_inputs->push_back({tensor_name, std::move(concated)});
  }

  return OkStatus();
}

Status BatchingSession::SplitOutputTensors(
    const TensorSignature& signature,
    const std::vector<Tensor>& combined_outputs,
    Batch<BatchingSessionTask>* batch) {
  DCHECK_GE(batch->num_tasks(), 1);
  if (batch->num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch->num_tasks());
  }

  std::vector<int64_t> task_sizes_plus_optional_padding;
  task_sizes_plus_optional_padding.reserve(batch->num_tasks());
  for (int i = 0; i < batch->num_tasks(); ++i) {
    task_sizes_plus_optional_padding.push_back(batch->task(i).zeroth_dim_size);
  }
  const int padding_size = RoundToLowestAllowedBatchSize(
                               options_.allowed_batch_sizes, batch->size()) -
                           batch->size();
  if (padding_size > 0) {
    task_sizes_plus_optional_padding.push_back(padding_size);
  }

  // For each output tensor name, a divided-up tensor with one entry per task.
  std::map<string, std::vector<Tensor>> split_tensors;

  // Populate 'split_tensors'.
  DCHECK_EQ(signature.output_tensors.size(), combined_outputs.size());
  if (combined_outputs.size() != signature.output_tensors.size()) {
    return errors::Internal("Wrong number of batched output tensors");
  }
  const std::vector<string> output_tensors(signature.output_tensors.begin(),
                                           signature.output_tensors.end());
  for (int i = 0; i < output_tensors.size(); ++i) {
    const string& tensor_name = output_tensors[i];
    const Tensor& tensor = combined_outputs[i];

    if (tensor.shape().dims() == 0) {
      return errors::FailedPrecondition(
          "Batched output tensor has 0 dimensions");
    }
    if (tensor.shape().dim_size(0) != batch->size() + padding_size) {
      return errors::FailedPrecondition(
          "Batched output tensor's 0th dimension does not equal the sum of the "
          "0th dimension sizes of the input tensors");
    }

    std::vector<Tensor> split_tensor;
    const Status split_status =
        tensor::Split(tensor, task_sizes_plus_optional_padding, &split_tensor);
    DCHECK(split_status.ok()) << split_status.ToString();
    if (!split_status.ok()) {
      return errors::Internal("Tensor split operation failed: ",
                              split_status.ToString());
    }
    DCHECK_EQ(split_tensor.size(), task_sizes_plus_optional_padding.size());
    if (split_tensor.size() != task_sizes_plus_optional_padding.size()) {
      return errors::Internal(
          "Tensor split operation did not work as expected; got ",
          split_tensor.size(), " splits; expected ",
          task_sizes_plus_optional_padding.size());
    }
    split_tensors[tensor_name] = std::move(split_tensor);
  }

  for (int i = 0; i < batch->num_tasks(); ++i) {
    BatchingSessionTask* task = batch->mutable_task(i);
    for (const string& tensor_name : *task->output_tensor_names) {
      auto split_tensor = split_tensors.find(tensor_name);
      DCHECK(split_tensor != split_tensors.end());
      if (split_tensor == split_tensors.end()) {
        return errors::Internal("Task does not conform to batch signature");
      }

      if (task->is_partial) {
        std::vector<Tensor>& tensor_vector =
            (*task->shared_outputs)[task->split_index];
        tensor_vector.push_back(std::move(split_tensor->second[i]));
      } else {
        task->outputs->push_back(std::move(split_tensor->second[i]));
      }
    }
  }
  // (Ignore a possible final split_tensors entry containing the padding.)

  return OkStatus();
}

Status BatchingSession::SplitRunMetadata(RunMetadata* batch_metadata,
                                         Batch<BatchingSessionTask>* batch) {
  if (batch->num_tasks() > 0) {
    if (batch_metadata->has_cost_graph()) {
      // Scale the batch aggregated to reflect the cost of an individual request
      // in the batch; this assumes all requests in a batch have an equal cost.
      for (size_t i = 0; i < batch_metadata->cost_graph().cost_size(); ++i) {
        CostGraphDef_AggregatedCost* cost =
            batch_metadata->mutable_cost_graph()->mutable_cost(i);
        const float agg_cost = cost->cost();
        cost->set_cost(agg_cost / static_cast<float>(batch->num_tasks()));
      }
    }

    for (size_t i = 0; i < batch->num_tasks(); ++i) {
      BatchingSessionTask* batching_session_task = batch->mutable_task(i);
      if (batching_session_task->is_partial) {
        // If 'is_partial', 'split_run_metadatas' is not nullptr and points
        // to a vector of size
        // 'batching_session_task->output_tensor_names->size'.
        (*batching_session_task
              ->split_run_metadatas)[batching_session_task->split_index] =
            *batch_metadata;
      } else {
        RunMetadata* run_metadata = batching_session_task->run_metadata;
        if (run_metadata != nullptr) {
          *run_metadata = *batch_metadata;
        }
      }
    }
  }

  return OkStatus();
}

void BatchingSession::ProcessBatch(
    const TensorSignature& signature,
    std::unique_ptr<Batch<BatchingSessionTask>> batch) {
  // As a possible performance optimization, consider overlapping the tensor
  // concatenation with waiting for the batch to close (i.e. do the
  // concatenation incrementally as tasks stream into the batch).
  batch->WaitUntilClosed();

  if (batch->empty()) {
    return;
  }

  const uint64_t dequeue_time_micros = EnvTime::NowMicros();

  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  Status status;
  auto finally = gtl::MakeCleanup([&status, &batch] {
    for (int i = 0; i < batch->num_tasks(); ++i) {
      BatchingSessionTask* task = batch->mutable_task(i);
      if (task->is_partial) {
        task->thread_safe_status->Update(status);
        task->done_callback();
      } else {
        *batch->mutable_task(i)->status = status;
        batch->mutable_task(i)->done->Notify();
      }
    }
  });

  // Make sure we have at least one task that hasn't exceeded its timeout from
  // queue time alone, and find the latest task deadline which we'll use for the
  // overall batch.
  bool all_tasks_timeout_exceeded = true;
  uint64_t batch_deadline_micros = 0;
  for (int i = 0; i < batch->num_tasks(); ++i) {
    const BatchingSessionTask& task = batch->task(i);
    // If the caller doesn't populate RunOptions, the timeout is 0 by default.
    // Interpret that as "no timeout" i.e. infinity.
    const int64_t task_timeout_micros =
        task.run_options.timeout_in_ms() <= 0
            ? INT_MAX
            : task.run_options.timeout_in_ms() * 1000;
    const uint64_t task_deadline_micros =
        task.enqueue_time_micros + task_timeout_micros;
    if (task_deadline_micros > dequeue_time_micros) {
      all_tasks_timeout_exceeded = false;
      if (task_deadline_micros > batch_deadline_micros) {
        batch_deadline_micros = task_deadline_micros;
      }
    }
    queuing_latency->GetCell(thread_pool_name_)
        ->Add(dequeue_time_micros - task.enqueue_time_micros);
  }
  if (all_tasks_timeout_exceeded) {
    status = Status(error::RESOURCE_EXHAUSTED,
                    "Run() timeout exceeded while waiting in batching queue");
    return;
  }

  RunOptions run_options = batch->task(0).run_options;
  if (batch_deadline_micros == INT_MAX) {
    run_options.set_timeout_in_ms(0);
  } else {
    run_options.set_timeout_in_ms(
        (batch_deadline_micros - dequeue_time_micros) / 1000);
  }

  std::vector<std::pair<string, Tensor>> merged_inputs;
  status = MergeInputTensors(signature, *batch, &merged_inputs);
  if (!status.ok()) {
    return;
  }

  absl::optional<thread::ThreadPoolOptions> thread_pool_options =
      batch->task(0).thread_pool_options;

  const std::vector<string> output_tensor_names(
      signature.output_tensors.begin(), signature.output_tensors.end());
  std::vector<Tensor> combined_outputs;
  RunMetadata run_metadata;
  // Because the wrapped session may not provide an implementation for
  // thread_pool_options, we need to invoke different Run() functions depending
  // on whether thread_pool_options is specified.
  if (thread_pool_options) {
    status = wrapped_->Run(run_options, merged_inputs, output_tensor_names,
                           {} /* target node names */, &combined_outputs,
                           &run_metadata, thread_pool_options.value());
  } else {
    status = wrapped_->Run(run_options, merged_inputs, output_tensor_names,
                           {} /* target node names */, &combined_outputs,
                           &run_metadata);
  }
  wrapped_run_count->GetCell()->IncrementBy(1);
  status.Update(SplitRunMetadata(&run_metadata, batch.get()));

  if (!status.ok()) {
    return;
  }

  status = SplitOutputTensors(signature, combined_outputs, batch.get());
}

// TODO(b/158393551):
// Share implementation between `SplitInputTask` here and
// `BatchResource::SplitInputTask` by refactoring and unifying the naming or
// type differences of data members.
Status SplitInputTask(
    std::unique_ptr<BatchingSessionTask>* input_task_ptr,
    int open_batch_remaining_slot, int max_batch_size,
    std::vector<std::unique_ptr<BatchingSessionTask>>* output_tasks) {
  BatchingSessionTask& input_task = *(*input_task_ptr);
  const int64_t input_task_size = input_task.size();

  DCHECK_GT(input_task_size, 0);

  // `split_task_done_callback` runs only after all split tasks are complete.
  std::function<void()> split_task_done_callback =
      [done_notification = input_task.done,
       shared_outputs = input_task.shared_outputs,
       shared_status = input_task.thread_safe_status,
       num_output = input_task.output_tensor_names->size(),
       outputs = input_task.outputs, status = input_task.status,
       run_metadata = input_task.run_metadata,
       split_run_metadatas = input_task.split_run_metadatas]() {
        auto finally = gtl::MakeCleanup([&] {
          *status = shared_status->status();
          done_notification->Notify();
        });

        // Some slices of tasks encounter errors, return early without
        // processing per-split result.
        if (!shared_status->status().ok()) {
          return;
        }

        for (int i = 0; i < num_output; ++i) {
          Tensor output_tensor;

          // Concat i-th tensor from each split into i-th tensor of output.
          std::vector<Tensor> to_concatenate;
          to_concatenate.reserve(shared_outputs->size());
          for (int j = 0; j < shared_outputs->size(); ++j) {
            to_concatenate.push_back(std::move((*shared_outputs)[j][i]));
          }
          const auto concat_status =
              tensor::Concat(to_concatenate, &output_tensor);
          if (!concat_status.ok()) {
            shared_status->Update(concat_status);
            return;
          }

          outputs->push_back(std::move(output_tensor));
        }

        // `cost_dimension_map` aggregates costs from all splits for each
        // dimension.
        absl::flat_hash_map<string, float> cost_dimension_map;
        for (const auto& split : *split_run_metadatas) {
          if (split.has_cost_graph()) {
            for (const auto& cost : split.cost_graph().cost()) {
              cost_dimension_map[cost.dimension()] += cost.cost();
            }
          }
        }

        *run_metadata = (*split_run_metadatas)[0];
        std::vector<string> cost_dimensions;
        for (const auto& cost_and_dimension :
             run_metadata->cost_graph().cost()) {
          cost_dimensions.push_back(cost_and_dimension.dimension());
        }
        run_metadata->mutable_cost_graph()->clear_cost();
        for (const auto& dimension : cost_dimensions) {
          const auto iter = cost_dimension_map.find(dimension);
          if (iter != cost_dimension_map.end()) {
            auto graph_cost = run_metadata->mutable_cost_graph()->add_cost();
            graph_cost->set_dimension(iter->first);
            graph_cost->set_cost(iter->second);
          }
        }
      };
  IncrementalBarrier barrier(split_task_done_callback);

  const internal::InputSplitMetadata input_split_metadata(
      input_task_size, open_batch_remaining_slot, max_batch_size);

  // Creates an array of int64_t from an array of int, since `tensor::Split`
  // requires an array of int64.
  const absl::FixedArray<int64_t> output_task_sizes(
      input_split_metadata.task_sizes().begin(),
      input_split_metadata.task_sizes().end());
  const int num_batches = output_task_sizes.size();

  input_task.shared_outputs->resize(num_batches);

  for (int i = 0; i < num_batches; ++i) {
    (*input_task.shared_outputs)[i].reserve(
        input_task.output_tensor_names->size());
  }

  input_task.split_run_metadatas->resize(num_batches);

  output_tasks->reserve(num_batches);
  for (int i = 0; i < num_batches; i++) {
    auto task = absl::make_unique<BatchingSessionTask>();
    task->enqueue_time_micros = input_task.enqueue_time_micros;
    task->run_options = input_task.run_options;
    task->zeroth_dim_size = output_task_sizes[i];
    // `task->owned_input` will be initialized separately out of this for-loop.
    task->output_tensor_names = input_task.output_tensor_names;

    task->owned_split_inputs =
        absl::make_unique<std::vector<std::pair<string, Tensor>>>();
    task->split_index = i;
    task->shared_outputs = input_task.shared_outputs;
    task->thread_safe_status = input_task.thread_safe_status;
    task->is_partial = true;
    task->done_callback = barrier.Inc();
    task->thread_pool_options = input_task.thread_pool_options;

    task->split_run_metadatas = input_task.split_run_metadatas;

    output_tasks->push_back(std::move(task));
  }

  const int num_input_tensors = input_task.inputs->size();

  // Splits each input tensor according to `output_task_sizes`, and
  // initializes input of `output_tasks` with split results.
  for (int i = 0; i < num_input_tensors; ++i) {
    std::vector<Tensor> split_tensors;
    const string& tensor_name = (*input_task.inputs)[i].first;
    const Tensor& input_tensor = (*input_task.inputs)[i].second;
    // TODO(b/158393551):
    // Figure out the optimal implementation of Split, by using
    // 'Tensor::Slice' and eliminating unnecessary memcpy as much as possible.
    const Status split_status =
        tensor::Split(input_tensor, output_task_sizes, &split_tensors);
    if (!split_status.ok()) {
      return errors::Internal(
          "When splitting input, Tensor split operation failed: ",
          split_status.ToString());
    }
    if (split_tensors.size() != output_task_sizes.size()) {
      return errors::Internal(
          "When splitting input, tensor split operation did not work as "
          "expected; got ",
          split_tensors.size(), " splits; expected ", output_task_sizes.size());
    }
    for (int j = 0; j < output_tasks->size(); ++j) {
      BatchingSessionTask& output_task = *((*output_tasks)[j]);
      output_task.owned_split_inputs->push_back(
          std::make_pair(tensor_name, split_tensors[j]));
    }
  }
  return OkStatus();
}

Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    BatchingSessionSchedulerCreator default_creator,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session) {
  std::unique_ptr<BatchingSession> internal_batching_session;
  TF_RETURN_IF_ERROR(BatchingSession::Create(
      options, std::move(session), signatures_with_scheduler_creators,
      default_creator, /*thread_pool_name=*/"", &internal_batching_session));
  *batching_session = std::move(internal_batching_session);
  return OkStatus();
}

Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session) {
  std::unique_ptr<BatchingSession> internal_batching_session;
  TF_RETURN_IF_ERROR(BatchingSession::Create(
      options, std::move(session), signatures_with_scheduler_creators,
      /*thread_pool_name=*/"", &internal_batching_session));
  *batching_session = std::move(internal_batching_session);
  return OkStatus();
}

Status CreateBasicBatchingSession(
    const BasicBatchScheduler<BatchingSessionTask>::Options& schedule_options,
    const BatchingSessionOptions& batching_session_options,
    const TensorSignature& signature, std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session) {
  const auto& allowed_batch_sizes =
      batching_session_options.allowed_batch_sizes;
  if (!allowed_batch_sizes.empty()) {
    if (schedule_options.enable_large_batch_splitting) {
      const int max_allowed_batch_size = allowed_batch_sizes.back();
      int32 last_size = 0;
      for (size_t i = 0; i < allowed_batch_sizes.size(); ++i) {
        const int32 size = allowed_batch_sizes.at(i);
        if (i > 0 && size <= last_size) {
          return errors::InvalidArgument(
              "allowed_batch_sizes entries must be monotonically increasing");
        }
        last_size = size;
      }
      if (max_allowed_batch_size > schedule_options.max_batch_size) {
        return errors::InvalidArgument(
            "Last entry in allowed_batch_sizes must be less than or equal to "
            "max_batch_size; last "
            "entry was ",
            max_allowed_batch_size, "; expected ",
            schedule_options.max_batch_size);
      }
      if (schedule_options.max_execution_batch_size != max_allowed_batch_size) {
        return errors::InvalidArgument(
            "Last entry in allowed_batch_sizes must be equal to "
            "max_execution_batch_size; last "
            "entry was ",
            max_allowed_batch_size, "; expected ",
            schedule_options.max_execution_batch_size);
      }
    } else if (allowed_batch_sizes.back() != schedule_options.max_batch_size) {
      // TODO(b/b/161641195):
      // Validate `allowed_batch_sizes` increase monotonically for non
      // large_batch_splitting case.
      return errors::InvalidArgument(
          "Last entry in allowed_batch_sizes must match max_batch_size; last "
          "entry was ",
          batching_session_options.allowed_batch_sizes.back(), "; expected ",
          schedule_options.max_batch_size);
    }
  }

  auto scheduler_creator =
      [schedule_options](
          std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
              process_batch_callback,
          std::unique_ptr<BatchScheduler<BatchingSessionTask>>*
              batch_scheduler) {
        std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>>
            basic_batch_scheduler;
        TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
            schedule_options, process_batch_callback, &basic_batch_scheduler));
        *batch_scheduler = std::move(basic_batch_scheduler);
        return OkStatus();
      };

  std::unique_ptr<BatchingSession> internal_batching_session;
  TF_RETURN_IF_ERROR(BatchingSession::Create(
      batching_session_options, std::move(session),
      {{signature, scheduler_creator}}, schedule_options.thread_pool_name,
      &internal_batching_session));
  *batching_session = std::move(internal_batching_session);
  return OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
