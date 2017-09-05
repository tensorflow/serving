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

#include "tensorflow/contrib/batching/basic_batch_scheduler.h"
#include "tensorflow/contrib/batching/batch_scheduler.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

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

  // If set to true, padding is performed for tensors of the same name
  // but with unequal dimensions (modulo zeroth dimension), so that
  // all tensors of the same name have equal dim sizes.
  // For each tensor its first element is used as padding value.
  //
  // For example:
  // given input tensors of shapes [1, 500, 101], [2, 300, 101], [1, 400, 101]
  // they will be padded to shapes [1, 500, 101], [2, 500, 101], [1, 500, 101].
  // Padding is not performed in zeroth dimension.
  //
  // Supported tensor datatypes:
  // DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16,
  // DT_UINT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128,
  // DT_STRING, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT16,
  // DT_QUINT16, DT_QINT32, DT_HALF, DT_RESOURCE.
  //
  // Supported ranks: from 1 to 6.
  //
  // This option is useful when using recurrent models(like LSTMs) with serving.
  // These models typically accept variable-length inputs and when
  // training them typical strategy is to save sequence lengths for decoding
  // and pad those variable-length dims to maximum in batch.
  // So, this option is used to achieve similar behavior
  // when serving with batching, it is assumed that sequence lengths
  // have already been saved.
  //
  // If tensors with the same name have different shapes
  // (modulo zeroth dimension) and this option is set to false,
  // then error Status will be returned.
  bool pad_variable_length_inputs = false;
};

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

// A convenience for using CreateBatchingSession() to create a
// BasicBatchScheduler for a single signature.
Status CreateBasicBatchingSession(
    const typename BasicBatchScheduler<BatchingSessionTask>::Options&
        schedule_options,
    const BatchingSessionOptions& batching_session_options,
    const TensorSignature& signature, std::unique_ptr<Session> session,
    std::unique_ptr<Session>* batching_session);

//////////
// Implementation details follow. API users need not read.

struct BatchingSessionTask : public BatchTask {
  ~BatchingSessionTask() override = default;
  size_t size() const override { return zeroth_dim_size; }

  // Fields populated when a task is received.
  uint64 enqueue_time_micros;
  RunOptions run_options;
  size_t zeroth_dim_size;
  const std::vector<std::pair<string, Tensor>>* inputs;
  const std::vector<string>* output_tensor_names;

  // Fields populated when a task is processed (as part of a batch).
  Notification* done;
  Status* status;
  std::vector<Tensor>* outputs;
  RunMetadata* run_metadata;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_SESSION_H_
