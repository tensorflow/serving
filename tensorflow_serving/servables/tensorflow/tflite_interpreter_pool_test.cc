/* Copyright 2021 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/servables/tensorflow/tflite_interpreter_pool.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace internal {

using tensorflow::gtl::ArraySlice;

constexpr char kParseExampleModel[] =
    "/servables/tensorflow/testdata/parse_example_tflite/00000123/"
    "model.tflite";

constexpr char kMobileNetModel[] =
    "/servables/tensorflow/testdata/mobilenet_v1_quant_tflite/00000123/"
    "model.tflite";

TEST(TfLiteInterpreterPool, CreateTfLiteInterpreterPoolTest) {
  string model_bytes;
  TF_ASSERT_OK(ReadFileToString(Env::Default(),
                                test_util::TestSrcDirPath(kParseExampleModel),
                                &model_bytes));
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(model_bytes.data()));
  int id = 0;
  bool use_batch_parallelism = false;
  int batch_pool_size = 1;
  bool run_in_caller = true;
  const tensorflow::SessionOptions options;
  std::unique_ptr<TfLiteInterpreterPool> pool;
  TF_ASSERT_OK(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
      *model, run_in_caller, use_batch_parallelism, batch_pool_size, id,
      options, pool));
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size);
  ASSERT_EQ(pool->Id(), id);
  ASSERT_EQ(pool->FixedBatchSize(), 1);  // batch_parallelism turned off.
  pool.reset();
  run_in_caller = false;
  use_batch_parallelism = true;
  TF_ASSERT_OK(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
      *model, run_in_caller, use_batch_parallelism, ++batch_pool_size, id,
      options, pool));
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size);
  ASSERT_EQ(pool->ThreadPool()->NumThreads(), batch_pool_size);
  ASSERT_EQ(pool->UseBatchParallelism(), use_batch_parallelism);
  pool.reset();
  TF_ASSERT_OK(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
      *model, run_in_caller, use_batch_parallelism, ++batch_pool_size, id,
      options, pool));
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size);
  ASSERT_EQ(pool->FixedBatchSize(), (kInitialBatchSize + 2) / batch_pool_size);
  pool.reset();
  use_batch_parallelism = false;
  run_in_caller = true;
  TF_ASSERT_OK(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
      *model, run_in_caller, use_batch_parallelism, batch_pool_size, id,
      options, pool));
  ASSERT_EQ(pool->NumInterpreters(), 1);
  ASSERT_EQ(pool->FixedBatchSize(), 1);  // batch_parallelism turned off.
  ASSERT_EQ(pool->ThreadPool(), nullptr);
  ASSERT_EQ(pool->UseBatchParallelism(), use_batch_parallelism);
  pool.reset();
  use_batch_parallelism = true;
  TF_ASSERT_OK(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
      *model, run_in_caller, use_batch_parallelism, batch_pool_size, id,
      options, pool));
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size + 1);
  ASSERT_EQ(pool->ThreadPool()->NumThreads(), batch_pool_size);
  pool.reset();
}

TEST(TfLiteSessionPool, CreateTfLiteSessionPoolTest) {
  string model_bytes;
  TF_ASSERT_OK(ReadFileToString(Env::Default(),
                                test_util::TestSrcDirPath(kParseExampleModel),
                                &model_bytes));
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(model_bytes.data()));
  int pool_size = 1;
  int batch_pool_size = 2;
  bool run_in_caller_thread = false;
  const tensorflow::SessionOptions options;
  std::unique_ptr<TfLiteSessionPool> session_pool;
  TF_ASSERT_OK(TfLiteSessionPool::CreateTfLiteSessionPool(
      model.get(), options, run_in_caller_thread, pool_size, batch_pool_size,
      session_pool));
  auto pool = session_pool->GetInterpreterPool();
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size);
  ASSERT_EQ(pool->Id(), 0);
  ASSERT_EQ(pool->FixedBatchSize(),
            (kInitialBatchSize + batch_pool_size - 1) / batch_pool_size);
  session_pool->ReturnInterpreterPool(std::move(pool));
  session_pool.reset();

  pool_size = 2;
  batch_pool_size = 2;
  run_in_caller_thread = false;
  TF_ASSERT_OK(TfLiteSessionPool::CreateTfLiteSessionPool(
      model.get(), options, run_in_caller_thread, pool_size, batch_pool_size,
      session_pool));

  pool = session_pool->GetInterpreterPool();
  ASSERT_EQ(pool->NumInterpreters(), batch_pool_size);
  ASSERT_EQ(pool->Id(), 1);
  ASSERT_EQ(pool->FixedBatchSize(), (kInitialBatchSize + 1) / 2);
  session_pool->ReturnInterpreterPool(std::move(pool));
  session_pool.reset();
}

TEST(TfLiteSessionPool, CreateTfLiteSessionPoolNotBatchParallelTest) {
  string model_bytes;
  TF_ASSERT_OK(ReadFileToString(Env::Default(),
                                test_util::TestSrcDirPath(kMobileNetModel),
                                &model_bytes));
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(model_bytes.data()));
  int pool_size = 1;
  int batch_pool_size = 1;
  bool run_in_caller_thread = false;
  const tensorflow::SessionOptions options;
  std::unique_ptr<TfLiteSessionPool> session_pool;
  TF_ASSERT_OK(TfLiteSessionPool::CreateTfLiteSessionPool(
      model.get(), options, run_in_caller_thread, pool_size, batch_pool_size,
      session_pool));

  auto pool = session_pool->GetInterpreterPool();
  ASSERT_EQ(pool->NumInterpreters(), 1);
  ASSERT_EQ(pool->Id(), 0);
  ASSERT_EQ(pool->FixedBatchSize(), 1);
  session_pool->ReturnInterpreterPool(std::move(pool));
  session_pool.reset();

  pool_size = 2;
  batch_pool_size = 2;
  run_in_caller_thread = false;
  TF_ASSERT_OK(TfLiteSessionPool::CreateTfLiteSessionPool(
      model.get(), options, run_in_caller_thread, pool_size, batch_pool_size,
      session_pool));

  pool = session_pool->GetInterpreterPool();
  ASSERT_EQ(pool->NumInterpreters(), 1);
  ASSERT_EQ(pool->Id(), 1);
  ASSERT_EQ(pool->FixedBatchSize(), 1);
  session_pool->ReturnInterpreterPool(std::move(pool));
  session_pool.reset();
}

int GetTensorSize(const TfLiteTensor* tflite_tensor) {
  int size = 1;
  for (int i = 0; i < tflite_tensor->dims->size; ++i) {
    size *= tflite_tensor->dims->data[i];
  }
  return size;
}

template <typename T>
std::vector<T> ExtractVector(const TfLiteTensor* tflite_tensor) {
  const T* v = reinterpret_cast<T*>(tflite_tensor->data.raw);
  return std::vector<T>(v, v + GetTensorSize(tflite_tensor));
}

template <>
std::vector<std::string> ExtractVector(const TfLiteTensor* tflite_tensor) {
  std::vector<std::string> out;
  for (int i = 0; i < tflite::GetStringCount(tflite_tensor); ++i) {
    auto ref = tflite::GetString(tflite_tensor, i);
    out.emplace_back(ref.str, ref.len);
  }
  return out;
}

TEST(TfLiteInterpreterWrapper, TfLiteInterpreterWrapperTest) {
  string model_bytes;
  TF_ASSERT_OK(ReadFileToString(Env::Default(),
                                test_util::TestSrcDirPath(kParseExampleModel),
                                &model_bytes));
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(model_bytes.data()));
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::custom::AddParseExampleOp(&resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter,
                                                         /*num_threads=*/1),
            kTfLiteOk);
  ASSERT_EQ(interpreter->inputs().size(), 1);
  const int idx = interpreter->inputs()[0];
  auto* tensor = interpreter->tensor(idx);
  ASSERT_EQ(tensor->type, kTfLiteString);
  int fixed_batch_size = 10;
  int actual_batch_size = 3;
  interpreter->ResizeInputTensor(idx, {fixed_batch_size});
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  auto interpreter_wrapper =
      std::make_unique<TfLiteInterpreterWrapper>(std::move(interpreter));
  interpreter_wrapper->SetMiniBatchSize(fixed_batch_size);
  ASSERT_EQ(interpreter_wrapper->GetMiniBatchSize(), fixed_batch_size);
  std::vector<tensorflow::tstring> data;
  std::vector<float> expected_floats;
  std::vector<std::string> expected_strs;
  for (int i = 0; i < actual_batch_size; ++i) {
    tensorflow::Example example;
    std::string str;
    auto* features = example.mutable_features();
    const float f = i % 2 == 1 ? 1.0 : -1.0;
    const std::string s = i % 2 == 1 ? "test" : "missing";
    expected_floats.push_back(f);
    expected_strs.push_back(s);
    (*features->mutable_feature())["x"].mutable_float_list()->add_value(
        expected_floats.back());
    (*features->mutable_feature())["y"].mutable_bytes_list()->add_value(
        expected_strs.back());
    example.SerializeToString(&str);
    data.push_back(str);
  }
  ASSERT_FALSE(interpreter_wrapper->SetStringData(data, tensor, -1) ==
               Status::OK());
  TF_ASSERT_OK(interpreter_wrapper->SetStringData(data, tensor, idx));
  auto wrapped = interpreter_wrapper->Get();
  ASSERT_EQ(wrapped->inputs().size(), 1);
  int input_idx = wrapped->inputs()[0];
  auto tflite_input_tensor = wrapped->tensor(input_idx);
  ASSERT_EQ(GetTensorSize(tflite_input_tensor), fixed_batch_size);
  ASSERT_EQ(tflite::GetStringCount(tflite_input_tensor), actual_batch_size);
  auto input_strs = ExtractVector<std::string>(tflite_input_tensor);
  EXPECT_THAT(input_strs, ::testing::ElementsAreArray(data));
  ASSERT_EQ(interpreter_wrapper->Invoke(), kTfLiteOk);
  const std::vector<int>& indices = wrapped->outputs();
  auto* tflite_tensor = wrapped->tensor(indices[0]);
  ASSERT_EQ(tflite_tensor->type, kTfLiteFloat32);
  ASSERT_EQ(GetTensorSize(tflite_tensor), fixed_batch_size);
  EXPECT_THAT(ArraySlice<float>(ExtractVector<float>(tflite_tensor).data(),
                                actual_batch_size),
              ::testing::ElementsAreArray(expected_floats));
  tflite_tensor = wrapped->tensor(indices[1]);
  ASSERT_EQ(tflite_tensor->type, kTfLiteString);
  ASSERT_EQ(GetTensorSize(tflite_tensor), fixed_batch_size);
  EXPECT_THAT(
      ArraySlice<std::string>(ExtractVector<std::string>(tflite_tensor).data(),
                              actual_batch_size),
      ::testing::ElementsAreArray(expected_strs));
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
