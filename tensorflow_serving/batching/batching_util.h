/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_BATCHING_BATCHING_UTIL_H_
#define TENSORFLOW_SERVING_BATCHING_BATCHING_UTIL_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace serving {

// For batch of inputs calculates maximum dim sizes across all tensors
// with the same name.
// These dim sizes are used later to calculate padding amount for each tensor.
// For example, for batch containing three tasks with the following inputs
// (instead of tensors there are their shapes):
//
// task1: {'tensor_a': [100, 500, 300], 'tensor_b': [100]}
// task2: {'tensor_a': [100, 200, 123], 'tensor_b': [100]}
// task3: {'tensor_a': [200, 100, 400], 'tensor_b': [200]}
//
// the following map will be generated:
// {'tensor_a': [200, 500, 400], 'tensor_b': [200]}
std::map<string, std::vector<int>> CalculateMaxDimSizes(
    const std::vector<std::vector<std::pair<string, Tensor>>>& batch);

// Pads tensor so that its shape becomes as specified in max_dim_sizes,
// except for zeroth dimension, which is left as is.
// First entry in max_dim_sizes is ignored.
// First element of a tensor is used as padding value.
// If tensor is empty, an error will be returned.
//
// For example given input tensor with shape [1, 2, 3, 4] and max_dim_sizes
// [1, 2, 5, 8] function produces padded_tensor of shape
// [1, 2, 5, 8], padded with tensor[0][0][0][0] element.
//
// Supported tensor datatypes:
// DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16,
// DT_UINT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128,
// DT_STRING, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT16,
// DT_QUINT16, DT_QINT32, DT_HALF, DT_RESOURCE.
//
// Supported tensor ranks: from 1 to 6.

Status AddPadding(const Tensor& tensor, absl::Span<const int> max_dim_sizes,
                  Tensor* padded_tensor);

// Returns the smallest entry in `allowed_batch_sizes` that is greater than or
// equal to `batch_size`. If `allowed_batch_sizes` is empty, simply returns
// `batch_size`.
int RoundToLowestAllowedBatchSize(absl::Span<const int> allowed_batch_sizes,
                                  int batch_size);

// Returns true iff all dims of shape1 are equal to dims of shape2 starting with
// the first (not zeroth) dimension.
// For example, for shapes [1, 2, 3] and [4, 2, 3] the result is true.
bool AreShapesEqualExceptZeroDim(const TensorShape& shape1,
                                 const TensorShape& shape2);

// Returns the first dimension size (batching dimension) of each tensor in
// `inputs`. If their first dimension sizes don't match, returns an error.
template <typename TensorList, typename DimFunc, typename DimSizeFunc>
Status ComputeTensorBatchSize(TensorList inputs, size_t* size, DimFunc dim_func,
                              DimSizeFunc dim_size_func) {
  if (inputs.empty()) {
    return errors::InvalidArgument(
        "Batching Run() must have at least one input tensor");
  }

  bool first = true;
  for (const auto& tensor : inputs) {
    if (dim_func(tensor) == 0) {
      return errors::InvalidArgument(
          "Batching Run() input tensors must have at least one "
          "dimension");
    }
    const size_t this_size = dim_size_func(tensor, 0);

    if (first) {
      *size = this_size;
      first = false;
    } else {
      if (this_size != *size) {
        return errors::InvalidArgument(
            "Batching Run() input tensors must have equal "
            "0th-dimension size");
      }
    }
  }
  return Status();
}

/***************** Below utilities are for monitoring purpose *****************/

// For all metrics: consider adding breakdowns based on model name or status if
// needed. Note that model name is not available as a session property or on any
// of the inputs currently.
template <typename BatchingTask>
void RecordPaddingSize(int32 padding_size, int32 execution_batch_size) {
  static const std::string batching_task_name = BatchingTask::Name();
  static auto* cell = tensorflow::monitoring::Sampler<1>::New(
      {absl::StrCat("/tensorflow/serving/", batching_task_name,
                    "/padding_size"),
       "Tracks the padding size distribution on batches.",
       "execution_batch_size"},
      // Exponential buckets [1*2^0, ..., 1*2^13, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 14));
  cell->GetCell(absl::StrCat(execution_batch_size))
      ->Add(static_cast<double>(padding_size));
}

template <typename BatchingTask>
void RecordInputBatchSize(int32 batch_size) {
  static const std::string batching_task_name = BatchingTask::Name();
  static auto* cell = tensorflow::monitoring::Sampler<0>::New(
      {absl::StrCat("/tensorflow/serving/", batching_task_name,
                    "/input_batch_size"),
       "Tracks the batch size distribution on the inputs."},
      // Exponential buckets [1*2^0, ..., 1*2^13, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 14));
  cell->GetCell()->Add(static_cast<double>(batch_size));
}

template <typename BatchingTask>
void RecordProcessedBatchSize(int32 batch_size) {
  static const std::string batching_task_name = BatchingTask::Name();
  static auto* cell = tensorflow::monitoring::Sampler<0>::New(
      {absl::StrCat("/tensorflow/serving/", batching_task_name,
                    "/processed_batch_size"),
       "Tracks the batch size distribution on processing."},
      // Exponential buckets [1*2^0, ..., 1*2^13, DBL_MAX].
      monitoring::Buckets::Exponential(1, 2, 14));
  cell->GetCell()->Add(static_cast<double>(batch_size));
}

}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_UTIL_H_
