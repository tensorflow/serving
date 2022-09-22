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

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/test_util/mock_file_probing_env.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::SetArgPointee;

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

  Example example_C(const int64_t value = 33) {
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
  const auto vec = tensor_.flat<tstring>();
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
  const auto vec = tensor_.flat<tstring>();
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
                EqualsProto(example_C().features().feature().at("c")));
    EXPECT_THAT(serialized_example.features().feature().at("b"),
                EqualsProto(example_B().features().feature().at("b")));
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
  const auto vec = tensor_.flat<tstring>();
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
  const auto vec = tensor_.flat<tstring>();
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

TEST(ExampleCountsTest, Simple) {
  using histogram::Histogram;

  const HistogramProto before_histogram =
      internal::GetExampleCounts()->GetCell("model-name")->value();
  const int before_count =
      internal::GetExampleCountTotal()->GetCell("model-name")->value();
  RecordRequestExampleCount("model-name", 3);
  const HistogramProto after_histogram =
      internal::GetExampleCounts()->GetCell("model-name")->value();
  const int after_count =
      internal::GetExampleCountTotal()->GetCell("model-name")->value();

  ASSERT_GE(before_histogram.bucket().size(), 3);
  ASSERT_GE(after_histogram.bucket().size(), 3);
  EXPECT_EQ(1, after_histogram.bucket(2) - before_histogram.bucket(2));
  EXPECT_EQ(3, after_count - before_count);
}

TEST(ModelSpecTest, NoOptional) {
  ModelSpec model_spec;
  MakeModelSpec("foo", /*signature_name=*/{}, /*version=*/{}, &model_spec);
  EXPECT_THAT(model_spec.name(), Eq("foo"));
  EXPECT_THAT(model_spec.signature_name(), ::testing::IsEmpty());
  EXPECT_FALSE(model_spec.has_version());
}

TEST(ModelSpecTest, OptionalSignature) {
  ModelSpec model_spec;
  MakeModelSpec("foo", /*signature_name=*/{"classify"}, /*version=*/{},
                &model_spec);
  EXPECT_THAT(model_spec.name(), Eq("foo"));
  EXPECT_THAT(model_spec.signature_name(), Eq("classify"));
  EXPECT_FALSE(model_spec.has_version());
}

TEST(ModelSpecTest, EmptySignature) {
  ModelSpec model_spec;
  MakeModelSpec("foo", /*signature_name=*/{""}, /*version=*/{1}, &model_spec);
  EXPECT_THAT(model_spec.name(), Eq("foo"));
  EXPECT_THAT(model_spec.signature_name(), Eq(kDefaultServingSignatureDefKey));
  EXPECT_THAT(model_spec.version().value(), Eq(1));
}

TEST(ModelSpecTest, OptionalVersion) {
  ModelSpec model_spec;
  MakeModelSpec("foo", /*signature_name=*/{}, /*version=*/{1}, &model_spec);
  EXPECT_THAT(model_spec.name(), Eq("foo"));
  EXPECT_THAT(model_spec.signature_name(), ::testing::IsEmpty());
  EXPECT_THAT(model_spec.version().value(), Eq(1));
}

TEST(ModelSpecTest, AllOptionalSet) {
  ModelSpec model_spec;
  MakeModelSpec("foo", /*signature_name=*/{"classify"}, /*version=*/{1},
                &model_spec);
  EXPECT_THAT(model_spec.name(), Eq("foo"));
  EXPECT_THAT(model_spec.signature_name(), Eq("classify"));
  EXPECT_THAT(model_spec.version().value(), Eq(1));
}

TEST(SignatureMethodNameCheckFeature, SetGet) {
  SetSignatureMethodNameCheckFeature(true);
  EXPECT_TRUE(GetSignatureMethodNameCheckFeature());

  SetSignatureMethodNameCheckFeature(false);
  EXPECT_FALSE(GetSignatureMethodNameCheckFeature());
}

TEST(ResourceEstimatorTest, EstimateResourceFromPathUsingDiskState) {
  const string export_dir = "/foo/bar";
  const string child = "child";
  const string child_path = io::JoinPath(export_dir, child);
  const double file_size = 100;

  // Set up the expectation that the directory contains exactly one child with
  // the given file size.
  test_util::MockFileProbingEnv env;
  EXPECT_CALL(env, FileExists(export_dir)).WillRepeatedly(Return(OkStatus()));
  EXPECT_CALL(env, GetChildren(export_dir, _))
      .WillRepeatedly(DoAll(SetArgPointee<1>(std::vector<string>({child})),
                            Return(OkStatus())));
  EXPECT_CALL(env, IsDirectory(child_path))
      .WillRepeatedly(Return(errors::FailedPrecondition("")));
  EXPECT_CALL(env, GetFileSize(child_path, _))
      .WillRepeatedly(DoAll(SetArgPointee<1>(file_size), Return(OkStatus())));

  ResourceAllocation actual;
  TF_ASSERT_OK(
      EstimateResourceFromPathUsingDiskState(export_dir, &env, &actual));

  ResourceAllocation expected =
      test_util::GetExpectedResourceEstimate(file_size);
  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST(GetMapKeysTest, GetKeys) {
  std::map<string, string> map = {std::pair<string, string>("key1", "value1"),
                                  std::pair<string, string>("key2", "value2")};
  const auto result = GetMapKeys(map);
  EXPECT_THAT(result, ::testing::UnorderedElementsAre("key1", "key2"));
}

TEST(SetDifferenceTEST, GetDiff) {
  std::set<string> result;
  EXPECT_THAT(SetDifference({"a", "b", "c"}, {"a", "b"}),
              ::testing::UnorderedElementsAre("c"));
  EXPECT_THAT(SetDifference({"a", "b", "c"}, {"a", "b", "d"}),
              ::testing::UnorderedElementsAre("c"));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
