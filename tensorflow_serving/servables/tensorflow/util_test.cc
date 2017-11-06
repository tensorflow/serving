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

#include "tensorflow_serving/servables/tensorflow/util.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;
using test_util::EqualsProto;

class InputUtilTest : public ::testing::Test {
 protected:
  // A few known Examples to use.
  Example example_A() {
    Feature feature;
    feature.mutable_int64_list()->add_value(11);
    Example example;
    (*example.mutable_features()->mutable_feature())["a"] = feature;
    return example;
  }

  Example example_B() {
    Feature feature;
    feature.mutable_int64_list()->add_value(22);
    Example example;
    (*example.mutable_features()->mutable_feature())["b"] = feature;
    return example;
  }

  Example example_C(const int64 value = 33) {
    Feature feature;
    feature.mutable_int64_list()->add_value(value);
    Example example;
    (*example.mutable_features()->mutable_feature())["c"] = feature;
    return example;
  }

  Input input_;
  Tensor tensor_;
};

TEST_F(InputUtilTest, Empty_KindNotSet) {
  const Status status = InputToSerializedExampleTensor(input_, &tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Input is empty"));
}

TEST_F(InputUtilTest, Empty_ExampleList) {
  input_.mutable_example_list();

  const Status status = InputToSerializedExampleTensor(input_, &tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Input is empty"));
}

TEST_F(InputUtilTest, Empty_ExampleListWithContext) {
  input_.mutable_example_list_with_context();

  const Status status = InputToSerializedExampleTensor(input_, &tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Input is empty"));
}

TEST_F(InputUtilTest, ExampleList) {
  *input_.mutable_example_list()->mutable_examples()->Add() = example_A();
  *input_.mutable_example_list()->mutable_examples()->Add() = example_B();

  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  EXPECT_EQ(2, tensor_.NumElements());
  const auto vec = tensor_.flat<string>();
  ASSERT_EQ(vec.size(), 2);
  Example serialized_example;
  ASSERT_TRUE(serialized_example.ParseFromString(vec(0)));
  EXPECT_THAT(serialized_example, EqualsProto(example_A()));
  ASSERT_TRUE(serialized_example.ParseFromString(vec(1)));
  EXPECT_THAT(serialized_example, EqualsProto(example_B()));
}

TEST_F(InputUtilTest, ExampleListWithContext) {
  auto* examples =
      input_.mutable_example_list_with_context()->mutable_examples();
  *examples->Add() = example_A();
  *examples->Add() = example_B();
  *input_.mutable_example_list_with_context()->mutable_context() = example_C();

  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  EXPECT_EQ(2, tensor_.NumElements());
  const auto vec = tensor_.flat<string>();
  ASSERT_EQ(vec.size(), 2);
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(0)));
    EXPECT_THAT(serialized_example.features().feature().at("c"), EqualsProto(
        example_C().features().feature().at("c")));
    EXPECT_THAT(serialized_example.features().feature().at("a"), EqualsProto(
        example_A().features().feature().at("a")));
  }
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(1)));
    EXPECT_THAT(serialized_example.features().feature().at("c"), EqualsProto(
        example_C().features().feature().at("c")));
    EXPECT_THAT(serialized_example.features().feature().at("b"), EqualsProto(
        example_B().features().feature().at("b")));
  }
}

// Tests whether individual examples do override the context.
TEST_F(InputUtilTest, ExampleListWithOverridingContext) {
  auto* examples =
      input_.mutable_example_list_with_context()->mutable_examples();
  *examples->Add() = example_A();
  *examples->Add() = example_C(64);
  *input_.mutable_example_list_with_context()->mutable_context() = example_C();

  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  EXPECT_EQ(2, tensor_.NumElements());
  const auto vec = tensor_.flat<string>();
  ASSERT_EQ(vec.size(), 2);
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(0)));
    EXPECT_THAT(serialized_example.features().feature().at("c"),
                EqualsProto(example_C().features().feature().at("c")));
    EXPECT_THAT(serialized_example.features().feature().at("a"),
                EqualsProto(example_A().features().feature().at("a")));
  }
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(1)));
    EXPECT_THAT(serialized_example.features().feature().at("c"),
                EqualsProto(example_C(64).features().feature().at("c")));
  }
}

TEST_F(InputUtilTest, ExampleListWithContext_NoContext) {
  auto* examples =
      input_.mutable_example_list_with_context()->mutable_examples();
  *examples->Add() = example_A();
  *examples->Add() = example_B();

  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  EXPECT_EQ(2, tensor_.NumElements());
  const auto vec = tensor_.flat<string>();
  ASSERT_EQ(vec.size(), 2);
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(0)));
    EXPECT_THAT(serialized_example, EqualsProto(example_A()));
  }
  {
    Example serialized_example;
    ASSERT_TRUE(serialized_example.ParseFromString(vec(1)));
    EXPECT_THAT(serialized_example, EqualsProto(example_B()));
  }
}

TEST_F(InputUtilTest, ExampleListWithContext_OnlyContext) {
  // Ensure that if there are no examples there is no output (even if the
  // context is specified).
  *input_.mutable_example_list_with_context()->mutable_context() = example_C();

  const Status status = InputToSerializedExampleTensor(input_, &tensor_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Input is empty"));
}

TEST_F(InputUtilTest, RequestNumExamplesStreamz) {
  Input input_1;
  *input_1.mutable_example_list()->mutable_examples()->Add() = example_A();
  *input_1.mutable_example_list()->mutable_examples()->Add() = example_B();
  Tensor tensor_1;
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_1, &tensor_1));
  EXPECT_EQ(2, tensor_1.NumElements());

  Input input_2;
  *input_2.mutable_example_list()->mutable_examples()->Add() = example_C();
  Tensor tensor_2;
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_2, &tensor_2));
  EXPECT_EQ(1, tensor_2.NumElements());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
