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

#include "tensorflow_serving/batching/batching_util.h"

#include <string>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace serving {

// Padding for one dimension of a tensor.
// pad_before is a number of values to add before elements in one dimension,
// pad_after is number of objects to add after.
// NOTE: fields are named this way because of Eigen::TensorMap::pad method.
// It requires padding to be an array of elements that have fields
// "first" and "second".
struct OneDimPadding {
  int64 first;   // pad before
  int64 second;  // pad after
};

// Constructs array of paddings, where:
// paddings[i].first - number of objects to add before elements in dimension i
// of given tensor,
// paddings[i].second - number of objects to add after elements in dimension i.
// This padding signature is used when performing internal padding with Eigen.
//
// When building paddings it is assumed that tensor needs to be padded
// after each dimension, so its shape matches max_dim_sizes,
// First entry of max_dim_sizes, which is maximum size of zeroth dimension,
// is ignored, because we don't perform padding in that dimension.
template <int num_dims>
Eigen::array<OneDimPadding, num_dims> CreatePadding(
    Tensor tensor, const std::vector<int>& max_dim_sizes) {
  Eigen::array<OneDimPadding, num_dims> padding;
  for (unsigned int i = 0; i < max_dim_sizes.size(); ++i) {
    if (i > 0 && max_dim_sizes[i] - tensor.dim_size(i) > 0) {
      padding[i] = {0, max_dim_sizes[i] - tensor.dim_size(i)};
    } else {
      padding[i] = {0, 0};
    }
  }
  return padding;
}

// Functor, which performs padding of given input tensor
// using specified padding signature.
// For example, given tensor of shape [1, 2, 3] and padding signature
// [[0, 0], [0, 2], [2, 2]]
// functor produces padded_tensor of shape [1, 4, 7].
template <typename T, int num_dims>
struct PadTensor {
  Status operator()(Tensor input,
                    const Eigen::array<OneDimPadding, num_dims>& padding,
                    Tensor* output) {
    TensorShape output_shape;
    for (int d = 0; d < num_dims; ++d) {
      // Pad before existing elements.
      const int32 before_d = padding[d].first;
      // Pad after existing elements.
      const int32 after_d = padding[d].second;
      output_shape.AddDim(before_d + input.dim_size(d) + after_d);
    }
    if (output_shape.num_elements() == input.NumElements()) {
      bool result = output->CopyFrom(input, output_shape);
      if (!result) {
        return errors::Internal("Couldn't create output.");
      }
      return Status::OK();
    }
    if (input.NumElements() < 1) {
      return errors::InvalidArgument(
          "Got empty tensor in batch of non-empty tensors.");
    }
    *output = Tensor(input.dtype(), output_shape);
    typename TTypes<T, num_dims>::Tensor inputs = input.tensor<T, num_dims>();
    T pad_value(input.flat<T>()(0));  // using existing values in padding
    output->tensor<T, num_dims>() = inputs.pad(padding, pad_value);
    return Status::OK();
  }
};

// Invokes padding procedure for specific tensor ranks.
// Only ranks from 1 to 6 are supported (like in PadOp).
template <typename T>
Status PadTensorOfSpecificType(const Tensor& tensor,
                               const std::vector<int>& max_dim_sizes,
                               Tensor* output_tensor) {
  int num_dims = tensor.dims();
  switch (num_dims) {
    case 1: {
      Eigen::array<OneDimPadding, 1> padding;
      padding = CreatePadding<1>(tensor, max_dim_sizes);
      PadTensor<T, 1> padding_functor = PadTensor<T, 1>();
      return padding_functor(tensor, padding, output_tensor);
    }
    case 2: {
      Eigen::array<OneDimPadding, 2> padding;
      padding = CreatePadding<2>(tensor, max_dim_sizes);
      PadTensor<T, 2> padding_functor = PadTensor<T, 2>();
      return padding_functor(tensor, padding, output_tensor);
    }
    case 3: {
      Eigen::array<OneDimPadding, 3> padding;
      padding = CreatePadding<3>(tensor, max_dim_sizes);
      PadTensor<T, 3> padding_functor = PadTensor<T, 3>();
      return padding_functor(tensor, padding, output_tensor);
    }
    case 4: {
      Eigen::array<OneDimPadding, 4> padding;
      padding = CreatePadding<4>(tensor, max_dim_sizes);
      PadTensor<T, 4> padding_functor = PadTensor<T, 4>();
      return padding_functor(tensor, padding, output_tensor);
    }
    case 5: {
      Eigen::array<OneDimPadding, 5> padding;
      padding = CreatePadding<5>(tensor, max_dim_sizes);
      PadTensor<T, 5> padding_functor = PadTensor<T, 5>();
      return padding_functor(tensor, padding, output_tensor);
    }
    case 6: {
      Eigen::array<OneDimPadding, 6> padding;
      padding = CreatePadding<6>(tensor, max_dim_sizes);
      PadTensor<T, 6> padding_functor = PadTensor<T, 6>();
      return padding_functor(tensor, padding, output_tensor);
    }
    default:
      // only ranks from 1 to 6 are supported
      // (like in tensorflow/core/kernels/pad_op.cc)
      return errors::InvalidArgument(
          "Only tensors with rank from 1 to 6 can be padded.");
  }
}

std::map<string, std::vector<int>> CalculateMaxDimSizes(
    const std::vector<std::vector<std::pair<string, Tensor>>>& batch) {
  std::map<string, std::vector<int>> max_dim_sizes;
  // Populate 'max_dim_sizes'
  // init
  const std::vector<std::pair<string, Tensor>>& task_inputs = batch[0];
  for (const auto& entry : task_inputs) {
    const string& tensor_name = entry.first;
    const Tensor& tensor = entry.second;
    max_dim_sizes[tensor_name] = std::vector<int>(tensor.dims(), 0);
  }
  // fill
  for (int i = 0; i < batch.size(); ++i) {
    const std::vector<std::pair<string, Tensor>>& task_inputs = batch[i];
    for (const auto& entry : task_inputs) {
      const string& tensor_name = entry.first;
      const Tensor& tensor = entry.second;

      std::vector<int>& max_dim_sizes_for_one_input =
          max_dim_sizes[tensor_name];
      for (int j = 0; j < tensor.dims(); ++j) {
        const int old_max_size = max_dim_sizes_for_one_input[j];
        if (tensor.shape().dim_size(j) > old_max_size) {
          max_dim_sizes_for_one_input[j] = tensor.shape().dim_size(j);
        }
      }
    }
  }
  return max_dim_sizes;
}

Status AddPadding(const Tensor& tensor, const std::vector<int>& max_dim_sizes,
                  Tensor* padded_tensor) {
  const DataType input_dtype = tensor.dtype();
  Status padding_status;
#define CASE(type)                                                           \
  case DataTypeToEnum<type>::value: {                                        \
    padding_status =                                                         \
        PadTensorOfSpecificType<type>(tensor, max_dim_sizes, padded_tensor); \
    break;                                                                   \
  }
  switch (input_dtype) {
    TF_CALL_ALL_TYPES(CASE);
    TF_CALL_QUANTIZED_TYPES(CASE);
    // quantized types macro doesn't include these types
    TF_CALL_quint16(CASE);
    TF_CALL_qint16(CASE);
    default:
      padding_status = errors::InvalidArgument("Unsupported type");
  }
#undef CASE
  return padding_status;
}
}  // namespace serving
}  // namespace tensorflow
