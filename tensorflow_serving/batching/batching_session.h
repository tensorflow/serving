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

// Constructs array of paddings where
// paddings[i].first - number of objects to add before elements in dimension i,
// paddings[i].second - number of objects to add after elements in dimension i.
// This padding signature is used when parforming internal padding with Eigen.
template<int num_dims>
std::array<std::pair<int32, int32>, num_dims> CreatePadding(Tensor tensor,
        const std::vector<int>& max_dim_sizes) {
    std::array<std::pair<int32, int32>, num_dims> padding;
  for (unsigned int i = 0; i < max_dim_sizes.size(); ++i) {
    if (max_dim_sizes[i] - tensor.dim_size(i) > 0) {
      padding[i] = {0, max_dim_sizes[i] - tensor.dim_size(i)};
    } else {
      padding[i] = {0, 0};
    }
  }
  return padding;
}

// Represents result of padding:
// padding_status - Status object, contains error if it occured during padding.
// padded_tensor - padded Tensor object.
struct PaddingResult {
  Tensor padded_tensor;
  Status padding_status;
};

// Functor, which performs padding of given input tensor
// using specified padding signature.
// For example, given tensor of shape [1, 2, 3] and padding signature
// [[0, 0], [0, 2], [2, 2]]
// functor produces PaddingResult with padded_tensor of shape [1, 4, 7].
template <typename T, int num_dims>
struct PadTensor {
  PaddingResult operator()(Tensor input,
      std::array<std::pair<int32, int32>, num_dims> padding) {
    TensorShape output_shape;
    for (int d = 0; d < num_dims; ++d) {
      // Pad before existing elements.
      const int32 before_d = padding[d].first;
      // Pad after existing elements.
      const int32 after_d = padding[d].second;
      output_shape.AddDim(before_d + input.dim_size(d) + after_d);
    }
    if (output_shape.num_elements() == input.NumElements()) {
      Tensor out;
      auto res = out.CopyFrom(input, output_shape);
      if (!res) {
        return {Tensor(), errors::Internal("Couldn't create output.")};
      }
      return {out, Status::OK()};
    }
    Tensor output(input.dtype(), output_shape);
    typename TTypes<T, num_dims>::Tensor inputs = input.tensor<T, num_dims>();
    T pad_value;
    output.tensor<T, num_dims>() = inputs.pad(padding, pad_value);
    return {output, Status::OK()};
  }
};
// functor specialization for scalars:
// we do not perform padding in that case.
template<typename T>
struct PadTensor<T, 0> {
  PaddingResult operator()(Tensor input,
      std::array<std::pair<int32, int32>, 0>  padding) {
    TensorShape output_shape;
    Tensor output(input.dtype(), output_shape);
    typename TTypes<T, 0>::Tensor inputs = input.tensor<T, 0>();
    output.tensor<T, 0>() = inputs;
    return {output, Status::OK()};
  }
};

// Invokes padding procedure for specific tensor ranks.
// Only ranks from 0 to 6 are supported (like in PadOp).
template<typename T>
PaddingResult PadTensorOfSpecificType(const Tensor& tensor,
                               const std::vector<int>& max_dim_sizes) {
  int num_dims = tensor.dims();
  PaddingResult res;
  switch (num_dims) {
    case 0: {
      std::array<std::pair<int32, int32>, 0> padding;
      padding = CreatePadding<0>(tensor, max_dim_sizes);
      PadTensor<T, 0> padding_functor = PadTensor<T, 0>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 1: {
      std::array<std::pair<int32, int32>, 1> padding;
      padding = CreatePadding<1>(tensor, max_dim_sizes);
      PadTensor<T, 1> padding_functor = PadTensor<T, 1>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 2: {
      std::array<std::pair<int32, int32>, 2> padding;
      padding = CreatePadding<2>(tensor, max_dim_sizes);
      PadTensor<T, 2> padding_functor = PadTensor<T, 2>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 3: {
      std::array<std::pair<int32, int32>, 3> padding;
      padding = CreatePadding<3>(tensor, max_dim_sizes);
      PadTensor<T, 3> padding_functor = PadTensor<T, 3>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 4: {
      std::array<std::pair<int32, int32>, 4> padding;
      padding = CreatePadding<4>(tensor, max_dim_sizes);
      PadTensor<T, 4> padding_functor = PadTensor<T, 4>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 5: {
      std::array<std::pair<int32, int32>, 5> padding;
      padding = CreatePadding<5>(tensor, max_dim_sizes);
      PadTensor<T, 5> padding_functor = PadTensor<T, 5>();
      res = padding_functor(tensor, padding);
      break;
    }
    case 6: {
      std::array<std::pair<int32, int32>, 6> padding;
      padding = CreatePadding<6>(tensor, max_dim_sizes);
      PadTensor<T, 6> padding_functor = PadTensor<T, 6>();
      res = padding_functor(tensor, padding);
      break;
    }
    default:
    // only ranks from 0 to 6 are supported
    // (like in tensorflow/core/kernels/pad_op.cc)
      res = {Tensor(), errors::InvalidArgument(
            "Only tensors with rank from 0 to 6 can be padded.")};
  }
  return res;
}

// For batch of inputs calculates maximum dim sizes across all tensors
// with the same name.
// These dim sizes are used later to calculate padding amount for each tensor.
// For zeroth dimension max_dim_size is set to zero,
// because padding is not performed in this dimension.
// For example, for batch containing three tasks with the following inputs
// (instead of tensors there are their shapes):
//
// task1: {'features': [100, 500, 300], 'true_labels': [100]}
// task2: {'features': [100, 200, 123], 'true_labels': [100]}
// task3: {'features': [200, 100, 400], 'true_labels': [200]}
//
// the following map will be generated:
// {'features': [0, 500, 400], 'true_labels': [0]}
std::map<string, std::vector<int>> CalculateMaxDimSizes(
    const Batch<BatchingSessionTask>& batch);

// Pads tensor so that its shape becomes as specified in max_dim_sizes,
// Its zeroth dimension is unchanged.
//
// For example given tensor with shape [1, 2, 3, 4] and max_dim_sizes
// [0, 2, 5, 8] function returns PaddingResult with padded_tensor of shape
// [1, 2, 5, 8].
PaddingResult AddPadding(const Tensor& tensor,
    const std::vector<int>& max_dim_sizes);

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
  // but with unequal dimensions, so that all tensors of the same name have
  // equal dim sizes.
  //
  // For example:
  // given input tensors of shapes [1, 500, 101], [2, 300, 101], [1, 400, 101]
  // they will be padded to shapes [1, 500, 101], [2, 500, 101], [1, 500, 101].
  //
  // This option is useful when using recurrent models(like LSTMs) with serving.
  // These models typically accept variable-length inputs and when
  // training them typical strategy is to save sequence lengths for decoding
  // and pad those variable-length dims to maximum in batch.
  // So, this flag is used to achieve similar behavior
  // when serving with batching, it is assumed that sequence lengths
  // have already been saved.
  bool pad_variable_length_inputs;
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
