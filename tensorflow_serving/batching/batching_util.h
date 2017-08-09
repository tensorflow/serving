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

#include "tensorflow/contrib/batching/batch_scheduler.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"



namespace tensorflow {
namespace serving {

// Constructs array of paddings, where:
// paddings[i].first - number of objects to add before elements in dimension i
// of given tensor,
// paddings[i].second - number of objects to add after elements in dimension i.
// This padding signature is used when performing internal padding with Eigen.
//
// When building paddings it is assumed that tensor needs to be padded
// after each dimension, so its shape matches max_dim_sizes,
// except zeroth dimension which is left as is.
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
      const std::array<std::pair<int32, int32>, num_dims>& padding) {
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
      const std::array<std::pair<int32, int32>, 0>&  padding) {
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
    const std::vector<std::vector<std::pair<string, Tensor>>>& batch);

// Pads tensor so that its shape becomes as specified in max_dim_sizes,
// Its zeroth dimension is unchanged.
//
// For example given tensor with shape [1, 2, 3, 4] and max_dim_sizes
// [0, 2, 5, 8] function returns PaddingResult with padded_tensor of shape
// [1, 2, 5, 8].
//
// Only tensors with rank from 0 to 6 are supported.
PaddingResult AddPadding(const Tensor& tensor,
    const std::vector<int>& max_dim_sizes);
}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_UTIL_H_

