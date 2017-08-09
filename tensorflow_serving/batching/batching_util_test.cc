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
  true_max_dim_sizes["x0"] = {0, 50, 30};
  true_max_dim_sizes["x1"] = {0, 101};
  EXPECT_EQ(max_dim_sizes, true_max_dim_sizes);
}

TEST(BatchingUtilTest, AddPadding) {
  std::vector<int> max_dim_sizes {0, 100, 200};
  std::vector<DataType> types {DT_FLOAT,
                               DT_DOUBLE,
                               DT_INT32,
                               DT_UINT8,
                               DT_INT16,
                               DT_INT8,
                               DT_STRING,
                               DT_COMPLEX64,
                               DT_INT64,
                               DT_BOOL,
                               DT_HALF,
                               DT_RESOURCE,
                               DT_COMPLEX128,
                               DT_UINT16};
  PaddingResult res;
  for (auto type : types) {
    Tensor tensor(type, {10, 20, 30});
    res = AddPadding(tensor, max_dim_sizes);
    EXPECT_EQ(res.padding_status, Status::OK());
    EXPECT_EQ(res.padded_tensor.shape(), TensorShape({10, 100, 200}));
  }
}

TEST(BatchingUtilTest, CreatePadding) {
  Tensor tensor(DT_FLOAT, {10, 20, 30});
  std::vector<int> max_dim_sizes {0, 100, 200};
  std::array<std::pair<int32, int32>, 3> true_paddings;
  true_paddings[0] = {0, 0};
  true_paddings[1] = {0, 80};
  true_paddings[2] = {0, 170};
  auto paddings = CreatePadding<3>(tensor, max_dim_sizes);
  EXPECT_EQ(paddings, true_paddings);
}

TEST(BatchingUtilTest, PadTensorOfSpecificType) {
  Tensor tensor(DT_FLOAT, {10, 20, 30});
  std::vector<int> max_dim_sizes {0, 100, 200};
  PaddingResult res;
  res = PadTensorOfSpecificType<float>(tensor, max_dim_sizes);
  EXPECT_EQ(res.padding_status, Status::OK());
  EXPECT_EQ(res.padded_tensor.shape(), TensorShape({10, 100, 200}));
  tensor = Tensor(DT_FLOAT, {10, 20, 30, 40, 50, 60, 70});
  res = PadTensorOfSpecificType<float>(tensor, max_dim_sizes);
  EXPECT_NE(res.padding_status, Status::OK());
}

TEST(BatchingUtilTest, PadTensor) {
  Tensor tensor(DT_FLOAT, {10, 20, 30});
  std::array<std::pair<int32, int32>, 3> paddings;
  paddings[0] = {0, 0};
  paddings[1] = {5, 10};
  paddings[2] = {10, 15};
  PaddingResult res;
  PadTensor<float, 3> padding_functor;
  res = padding_functor(tensor, paddings);
  EXPECT_EQ(res.padding_status, Status::OK());
  EXPECT_EQ(res.padded_tensor.shape(), TensorShape({10, 35, 55}));
  Tensor scalar(DT_FLOAT, {});
  PadTensor<float, 0> scalar_padding_functor;
  res = scalar_padding_functor(scalar,
      std::array<std::pair<int32, int32>, 0>());
  EXPECT_EQ(res.padding_status, Status::OK());
  EXPECT_EQ(res.padded_tensor.shape(), scalar.shape());
}
}  // namespace
}  // namespace serving
}  // namespace tensorflow
