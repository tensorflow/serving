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

  Example example_C() {
    Feature feature;
    feature.mutable_int64_list()->add_value(33);
    Example example;
    (*example.mutable_features()->mutable_feature())["c"] = feature;
    return example;
  }

  Input input_;
  Tensor tensor_;
};

TEST_F(InputUtilTest, Empty_KindNotSet) {
  EXPECT_EQ(0, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  test::ExpectTensorEqual<string>(test::AsTensor<string>({}, TensorShape({0})),
                                  tensor_);
}

TEST_F(InputUtilTest, Empty_ExampleList) {
  input_.mutable_example_list();

  EXPECT_EQ(0, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  test::ExpectTensorEqual<string>(test::AsTensor<string>({}, TensorShape({0})),
                                  tensor_);
}

TEST_F(InputUtilTest, Empty_ExampleListWithContext) {
  input_.mutable_example_list_with_context();

  EXPECT_EQ(0, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  test::ExpectTensorEqual<string>(test::AsTensor<string>({}, TensorShape({0})),
                                  tensor_);
}

TEST_F(InputUtilTest, ExampleList) {
  *input_.mutable_example_list()->mutable_examples()->Add() = example_A();
  *input_.mutable_example_list()->mutable_examples()->Add() = example_B();

  EXPECT_EQ(2, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
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

  EXPECT_EQ(2, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
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

TEST_F(InputUtilTest, ExampleListWithContext_NoContext) {
  auto* examples =
      input_.mutable_example_list_with_context()->mutable_examples();
  *examples->Add() = example_A();
  *examples->Add() = example_B();

  EXPECT_EQ(2, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
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

  EXPECT_EQ(0, NumInputExamples(input_));
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_, &tensor_));
  test::ExpectTensorEqual<string>(test::AsTensor<string>({}, TensorShape({0})),
                                  tensor_);
}

TEST_F(InputUtilTest, RequestNumExamplesStreamz) {
  Input input_1;
  *input_1.mutable_example_list()->mutable_examples()->Add() = example_A();
  *input_1.mutable_example_list()->mutable_examples()->Add() = example_B();
  EXPECT_EQ(2, NumInputExamples(input_1));
  Tensor tensor_1;
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_1, &tensor_1));

  Input input_2;
  *input_2.mutable_example_list()->mutable_examples()->Add() = example_C();
  EXPECT_EQ(1, NumInputExamples(input_2));
  Tensor tensor_2;
  TF_ASSERT_OK(InputToSerializedExampleTensor(input_2, &tensor_2));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
