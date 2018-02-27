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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

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

Status AddPadding(const Tensor& tensor, const std::vector<int>& max_dim_sizes,
                  Tensor* padded_tensor);
}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_UTIL_H_
