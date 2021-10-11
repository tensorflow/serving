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

// A library for wrapping a tensorflow session such that Run() calls get
// scheduled in batches, using a batch scheduler of your choice.

#ifndef TENSORFLOW_SERVING_BATCHING_BATCHING_SESSION_H_
#define TENSORFLOW_SERVING_BATCHING_BATCHING_SESSION_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/batching/batching_options.h"
#include "tensorflow_serving/batching/threadsafe_status.h"

namespace tensorflow {
namespace serving {

// The batch scheduler task type used for batching sessions, for use in batch
// scheduler template parameters, e.g. BasicBatchScheduler<BatchingSessionTask>.
struct BatchingSessionTask;

// A function to construct a batch scheduler for BatchingSessionTasks from a
// process-batch callback.
using BatchingSessionSchedulerCreator = std::function<Status(
    std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>,
    std::unique_ptr<BatchScheduler<BatchingSessionTask>>*)>;

// The signature associated with a Session::Run() call, in terms of input and
// output tensor names (with the order in which the tensors are listed factored
// out). (Note that 'target_node_names' are not supported in batching sessions.)
struct TensorSignature {
  std::set<string> input_tensors;
  std::set<string> output_tensors;
};

// Constructs a TensorSignature for a given SignatureDef.
TensorSignature TensorSignatureFromSignatureDef(
    const SignatureDef& signature_def);

// Constructs a TensorSignature for a given set of SignatureDefs. The resulting
// TensorSignature represents the Session::Run() arguments that would be used
// when issuing a single Run() call that exercises the signature defs jointly.
//
// For example, say there's a graph that takes 'input' and transforms it into
// 'predicted_label' and 'confidence_score'. Suppose SignatureDef 1 requests
// only 'predicted_label' as output, and SignatureDef 2 requests only
// 'confidence_score'. A joint TensorSignature would feed 'input' and receive
// both 'predicted_label' and 'confidence_score' as output, in a single Run()
// invocation.
TensorSignature TensorSignatureFromSignatureDefs(
    const std::vector<SignatureDef>& signature_defs);

// A signature paired with a lambda to create a batch scheduler for Run() calls
// matching the signature.
struct SignatureWithBatchingSessionSchedulerCreator {
  TensorSignature signature;
  BatchingSessionSchedulerCreator scheduler_creator;
};

// Options for batching tensorflow Sessions; see the Create*() functions below.
using BatchingSessionOptions = BatchingOptions;

// Wraps a session in a new session that automatically batches Run() calls.
// Uses one batcher for each distinct Run() signature supported. In addition to
// a session to wrap, takes a list of signature/BatchingSessionSchedulerCreator
// pairs. (The number of supported signatures is typically small, and often just
// a single one.)
//
// The wrapped session only batches Run() calls that conform to one of the
// specified signatures and leave 'target_node_names' empty. Other Run() calls
// are executed in-line without batching, and may harm performance. (Extra-
// signature Run() support is intended primarily for debugging and diagnostics.)
//
// For batched calls, it is assumed that the outermost (0th) dimension of each
// input and output tensor is the batch-size dimension. All input tensors must
// have the same 0th-dimension size B; the produced output tensors are also
// assumed to have 0th-dimension size B.
//
// IMPORTANT: Each call to Session::Run() is synchronous, and blocks waiting for
// other Run() calls with the same signature to merge with to form a large
// batch. Consequently, to achieve good throughput we recommend setting the
// number of client threads that call Session::Run() equal to about twice the
// sum over all signatures of the maximum batch size.
//
// Example usage, for the common case of a single signature:
//
// BatchingSessionOptions options = ...;
// auto scheduler_creator = [schedule_options, retry_options](
//     std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
//         process_batch_callback,
//     std::unique_ptr<BatchScheduler<BatchingSessionTask>>* batch_scheduler) {
//   std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>> scheduler;
//   TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
//       schedule_options, process_batch_callback, &scheduler));
//   std::unique_ptr<BatchSchedulerRetrier<BatchingSessionTask>> retrier;
//   TF_RETURN_IF_ERROR(BatchSchedulerRetrier<BatchingSessionTask>::Create(
//       retry_options, std::move(scheduler), &retrier));
//   *batch_scheduler = std::move(retrier);
//   return Status::OK();
// };
// std::unique_ptr<Session> batching_session;
// TF_CHECK_OK(CreateBatchingSession(options, {{signature, scheduler_creator}},
//     std::move(session), &batching_session));
//
Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

// Same as above but allows for a default scheduler creator for which signatures
// that don't match a supplied value during run time can still use batching.
Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    const std::vector<SignatureWithBatchingSessionSchedulerCreator>&
        signatures_with_scheduler_creators,
    BatchingSessionSchedulerCreator default_creator,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

// A convenience for using CreateBatchingSession() to create a
// BasicBatchScheduler for a single signature.
Status CreateBasicBatchingSession(
    const typename BasicBatchScheduler<BatchingSessionTask>::Options&
        schedule_options,
    const BatchingSessionOptions& batching_session_options,
    const TensorSignature& signature, std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

// The default implementation of
// `BasicBatchScheduler::Options.split_input_task_func` if corresponding batch
// scheduler for a batching session sets
// `BasicBatchScheduler::Options.enable_large_batch_splitting` to true.
Status SplitInputTask(
    std::unique_ptr<BatchingSessionTask>* input_task_ptr,
    int open_batch_remaining_slot, int max_batch_size,
    std::vector<std::unique_ptr<BatchingSessionTask>>* output_tasks);

//////////
// Implementation details follow. API users need not read.

struct BatchingSessionTask : public BatchTask {
  ~BatchingSessionTask() override = default;
  size_t size() const override { return zeroth_dim_size; }

  // For monitoring purpose.
  static std::string Name() { return "batching_session"; }

  // Fields populated when a task is received.
  uint64_t enqueue_time_micros;
  RunOptions run_options;
  size_t zeroth_dim_size;
  const std::vector<std::pair<string, Tensor>>* inputs;
  const std::vector<string>* output_tensor_names;

  // Fields populated when a task is processed (as part of a batch), and
  // returned by BatchingSession when a task is complete.
  Notification* done;
  Status* status;
  std::vector<Tensor>* outputs;
  RunMetadata* run_metadata;
  absl::optional<thread::ThreadPoolOptions> thread_pool_options;

  // Fields populated when a task is processed (as part of a batch), and
  // substantially used in the intermediate stage if a task is a slice of
  // input task (i.e., is_partial=true).
  bool is_partial = false;
  // 'owned_split_inputs' stores pairs of tensor names and input tensors
  // if 'is_partial' = true.
  std::unique_ptr<std::vector<std::pair<string, Tensor>>> owned_split_inputs;
  // The index of this split, along the 0-th dimension of input from op
  // invocation.
  int split_index = 0;
  std::function<void()> done_callback;
  typedef std::vector<std::vector<Tensor>> TensorMatrix;
  // For shared_ptr objects, ownership shared by:
  // 1) each split of task (to fill one row in this matrix)
  // and
  // 2) callback that runs to merge output of individual splits for an op
  // invocation, after all splits complete.
  // Two-dimensional tensor matrix,
  std::shared_ptr<TensorMatrix> shared_outputs;
  // 'status' records error (could be from any split) if at least one split
  // returns error, OK otherwise.
  std::shared_ptr<ThreadSafeStatus> thread_safe_status;
  // 'split_run_metadatas' records `run_metadata` of each split.
  std::shared_ptr<std::vector<RunMetadata>> split_run_metadatas;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_SESSION_H_
