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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow {
namespace serving {

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
        int old_max_size = max_dim_sizes_for_one_input[j];
        if (tensor.shape().dim_size(j) > old_max_size) {
          max_dim_sizes_for_one_input[j] = tensor.shape().dim_size(j);
        }
      }
      max_dim_sizes_for_one_input[0] = 0;  // no need to pad in zeroth dimension
    }
  }
  return max_dim_sizes;
}

PaddingResult AddPadding(const Tensor& tensor,
                         const std::vector<int>& max_dim_sizes) {
  DataType input_dtype = tensor.dtype();
  PaddingResult res;
#define CASE(type) \
        case DataTypeToEnum<type>::value: { \
          res = PadTensorOfSpecificType<type>(tensor, max_dim_sizes); \
          break; \
        }
       switch (input_dtype) {
         TF_CALL_ALL_TYPES(CASE);
         default:
           res = {Tensor(), errors::InvalidArgument("Unsupported type")};
       }
#undef CASE
  return res;
}

}  // namespace serving
}  // namespace tensorflow
