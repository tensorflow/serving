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

#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/batching/basic_batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler.h"

namespace tensorflow {
namespace serving {

// The batch scheduler task type used for batching sessions, for use in batch
// scheduler template parameters, e.g. BasicBatchScheduler<BatchingSessionTask>.
struct BatchingSessionTask;

// Options for batching tensorflow Sessions; see the Create*() functions below.
struct BatchingSessionOptions {
  // If set, restricts the allowed tensor batch sizes.
  //
  // When the batch scheduler forms a batch of size N, the batch size is rounded
  // up to the smallest value M in 'allowed_batch_sizes' s.t. M >= N. The
  // tensors submitted to the underlying Session are padded with M-N repeats of
  // one of the first N entries (i.e. a guaranteed valid entry). The last M-N
  // entries of the output tensors are ignored.
  //
  // This option is useful when the underlying platform has some per-batch-size
  // overhead, to limit the number of distinct batch sizes that can occur. It
  // may be sensible to use an exponential sequence e.g. [8, 16, 32, ...,
  // max_batch_size], a linear one e.g. [100, 200, 300, ..., max_batch_size], or
  // perhaps a hybrid e.g. [8, 16, 32, 64, 100, 200, 300, ..., max_batch_size].
  //
  // IMPORTANT: The entries must be in increasing order.
  //
  // IMPORTANT: The final entry in 'allowed_batch_sizes' must equal the maximum
  //            batch size parameter supplied to the batch scheduler.
  //
  // If left empty, no rounding/padding is performed.
  std::vector<int> allowed_batch_sizes;
};

// Wraps a session in a new session that automatically batches Run() calls.
// In addition to a session to wrap, takes a function that constructs a batch
// scheduler given a process-batch callback.
//
// The wrapped session only supports Run(), with the following restrictions:
//  - 'target_node_names' must be empty
//  - 'inputs' must not be empty
//  - all calls supply the same input tensor names in 'inputs'
//  - all calls supply the same output tensor names in 'output_tensor_names'
//
// It is assumed that the outermost (0th) dimension of each input and output
// tensor is the batch-size dimension. All input tensors must have the same 0th-
// dimension size B; the produced output tensors are also assumed to have 0th-
// dimension size B.
//
// IMPORTANT: Each call to Session::Run() is synchronous, and blocks waiting for
// other Run() calls to merge with to form a large batch. Consequently, to
// achieve good throughput we recommend setting the number of client threads
// that call Session::Run() equal to about twice the maximum batch size.
//
// Example usage:
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
// TF_CHECK_OK(CreateBatchingSession(options, scheduler_creator,
//     std::move(session), &batching_session));
//
Status CreateBatchingSession(
    const BatchingSessionOptions& options,
    std::function<
        Status(std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>,
               std::unique_ptr<BatchScheduler<BatchingSessionTask>>*)>
        batch_scheduler_creator,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

// A convenience for using CreateBatchingSession() to create a
// BasicBatchScheduler wrapped in a retrier.
Status CreateRetryingBasicBatchingSession(
    const typename BasicBatchScheduler<BatchingSessionTask>::Options&
        schedule_options,
    const typename BatchSchedulerRetrier<BatchingSessionTask>::Options&
        retry_options,
    const BatchingSessionOptions& batching_session_options,
    std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

//////////
// Implementation details follow. API users need not read.

struct BatchingSessionTask : public BatchTask {
  ~BatchingSessionTask() override = default;
  size_t size() const override { return zeroth_dim_size; }

  // Fields populated when a task is received.
  size_t zeroth_dim_size;
  const std::vector<std::pair<string, Tensor>>* inputs;
  const std::vector<string>* output_tensor_names;

  // Fields populated when a task is processed (as part of a batch).
  Notification* done;
  Status* status;
  std::vector<Tensor>* outputs;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_SESSION_H_
