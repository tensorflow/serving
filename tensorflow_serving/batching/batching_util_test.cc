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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// Creates vector of pairs (tensor_name, tensor_value), where tensors
// have shapes as specified in shapes.
// Tensor with shape shapes[i] has tensor_name "x" + std::to_string(i).
std::vector<std::pair<string, Tensor>> CreateInputsWithTensorShapes(
    const std::vector<TensorShape>& shapes) {
  std::vector<std::pair<string, Tensor>> inputs;
  for (int i = 0; i < shapes.size(); ++i) {
    inputs.push_back({"x" + std::to_string(i), Tensor(DT_FLOAT, shapes[i])});
  }
  return inputs;
}

TEST(BatchingUtilTest, CalculateMaxDimSizes) {
  const std::vector<TensorShape> shapes1{{10, 20, 30}, {10, 100}};
  std::vector<std::pair<string, Tensor>> inputs1 =
      CreateInputsWithTensorShapes(shapes1);
  const std::vector<TensorShape> shapes2{{20, 50, 15}, {20, 101}};
  std::vector<std::pair<string, Tensor>> inputs2 =
      CreateInputsWithTensorShapes(shapes2);
  std::vector<std::vector<std::pair<string, Tensor>>> batch{inputs1, inputs2};
  std::map<string, std::vector<int>> max_dim_sizes =
      CalculateMaxDimSizes(batch);
  EXPECT_THAT(max_dim_sizes,
              UnorderedElementsAre(Pair("x0", ElementsAre(20, 50, 30)),
                                   Pair("x1", ElementsAre(20, 101))));
}

TEST(BatchingUtilTest, AddPadding) {
  const std::vector<int> max_dim_sizes{20, 100, 200};
  const std::vector<DataType> types{
      DT_FLOAT,      DT_DOUBLE, DT_INT32,  DT_UINT8,   DT_INT16,
      DT_UINT16,     DT_INT8,   DT_STRING, DT_BOOL,    DT_COMPLEX64,
      DT_COMPLEX128, DT_INT64,  DT_QINT8,  DT_QUINT8,  DT_QINT16,
      DT_QUINT16,    DT_QINT32, DT_HALF,   DT_RESOURCE};
  Status padding_status;
  for (DataType type : types) {
    Tensor tensor(type, {10, 20, 30});
#define INIT_TYPE(T)                      \
  if (type == DataTypeToEnum<T>::value) { \
    tensor.flat<T>().setConstant(T());    \
  }
    TF_CALL_ALL_TYPES(INIT_TYPE);
    TF_CALL_QUANTIZED_TYPES(INIT_TYPE);
    // quantized types macro doesn't include these types
    TF_CALL_quint16(INIT_TYPE);
    TF_CALL_qint16(INIT_TYPE);
#undef INIT_TYPE
    Tensor padded_tensor;
    padding_status = AddPadding(tensor, max_dim_sizes, &padded_tensor);
    ASSERT_EQ(Status::OK(), padding_status);
    EXPECT_EQ(TensorShape({10, 100, 200}), padded_tensor.shape());
  }
}

TEST(BatchingUtilTest, AddPaddingTensorWithUnsupportedRank) {
  const std::vector<int> max_dim_sizes{1, 1, 1, 1, 1, 1, 1};
  const Tensor tensor(DT_FLOAT, {1, 1, 1, 1, 1, 1, 1});
  Tensor padded_tensor;
  ASSERT_EQ(errors::InvalidArgument(
                "Only tensors with rank from 1 to 6 can be padded."),
            AddPadding(tensor, max_dim_sizes, &padded_tensor));
}
}  // namespace
}  // namespace serving
}  // namespace tensorflow
