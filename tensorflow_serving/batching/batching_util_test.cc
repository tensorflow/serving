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



#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace {
std::vector<std::pair<string, Tensor>> CreateInputsWithTensorShapes(
        const std::vector<TensorShape>& shapes) {
  std::vector<std::pair<string, Tensor>> inputs;
  for (int i = 0; i < shapes.size(); ++i) {
    inputs.push_back({"x" + std::to_string(i), Tensor(DT_FLOAT, shapes[i])});
  }
  return inputs;
}


TEST(BatchingUtilTest, CalculateMaxDimSizes) {
  std::vector<TensorShape> shapes1;
  shapes1.push_back({10, 20, 30});  // x0
  shapes1.push_back({10, 100});  // x1
  std::vector<std::pair<string, Tensor>> inputs1 =
    CreateInputsWithTensorShapes(shapes1);
  std::vector<TensorShape> shapes2;
  shapes2.push_back({20, 50, 15});  // x0
  shapes2.push_back({20, 101});  // x1
  std::vector<std::pair<string, Tensor>> inputs2 =
    CreateInputsWithTensorShapes(shapes2);
  std::vector<std::vector<std::pair<string, Tensor>>> batch {inputs1, inputs2};
  std::map<string, std::vector<int>> max_dim_sizes =
    CalculateMaxDimSizes(batch);
  std::map<string, std::vector<int>> true_max_dim_sizes;
  true_max_dim_sizes["x0"] = {20, 50, 30};
  true_max_dim_sizes["x1"] = {20, 101};
  EXPECT_EQ(max_dim_sizes, true_max_dim_sizes);
}

TEST(BatchingUtilTest, AddPadding) {
  std::vector<int> max_dim_sizes {20, 100, 200};
  std::vector<DataType> types {DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8,
      DT_INT16, DT_UINT16, DT_INT8, DT_STRING, DT_COMPLEX64, DT_COMPLEX128,
      DT_INT64, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT16,
      DT_QUINT16, DT_QINT32, DT_HALF, DT_RESOURCE};
  Status padding_status;
  for (auto type : types) {
    Tensor tensor(type, {10, 20, 30});
    Tensor padded_tensor;
    padding_status = AddPadding(tensor, max_dim_sizes, &padded_tensor);
    EXPECT_EQ(padding_status, Status::OK());
    EXPECT_EQ(padded_tensor.shape(), TensorShape({10, 100, 200}));
  }
}

TEST(BatchingUtilTest, AddPaddingTensorWithUnsupportedRank) {
  std::vector<int> max_dim_sizes {20, 100, 200, 300, 400, 500, 600};
  Tensor tensor(DT_FLOAT, {10, 20, 30, 40, 50, 60, 70});
  Tensor padded_tensor;
  Status padding_status = AddPadding(tensor, max_dim_sizes, &padded_tensor);
  EXPECT_EQ(padding_status, errors::InvalidArgument(
            "Only tensors with rank from 0 to 6 can be padded."));
}

TEST(BatchingUtilTest, AddPaddingScalar) {
  std::vector<int> max_dim_sizes;
  Tensor scalar(DT_FLOAT, {});
  Tensor padded_tensor;
  Status padding_status = AddPadding(scalar, max_dim_sizes, &padded_tensor);
  EXPECT_EQ(padding_status, Status::OK());
  EXPECT_EQ(padded_tensor.shape(), scalar.shape());
}
}  // namespace
}  // namespace serving
}  // namespace tensorflow
