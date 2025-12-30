/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/batching/tfrt_saved_model_with_batching.h"

#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow_serving/batching/batching_util.h"
#include "tensorflow_serving/servables/tensorflow/test_util/mock_tfrt_saved_model.h"

namespace tensorflow {
namespace serving {

namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Invoke;
using ::testing::Return;

constexpr char kFunctionOne[] = "func1";
constexpr char kFunctionTwo[] = "func2";
constexpr char kUnknownFunction[] = "unknown_func";
const tfrt::internal::Signature signature;

// TODO(b/168220822): Consider declaring a TensorMatcher for more
// exhaustive error messages.
MATCHER_P(MatchesTensor, p, "") {
  const Tensor &x = arg;
  const Tensor &y = *p;
  const float *Tx = x.unaligned_flat<float>().data();
  const float *Ty = y.unaligned_flat<float>().data();
  auto size = x.NumElements();
  for (decltype(size) i = 0; i < size; ++i) {
    if (Tx[i] != Ty[i]) return false;
  }
  return true;
}

MATCHER_P2(TFStatusIs, error_code, partial_error_message, "") {
  return arg.code() == error_code &&
         absl::StrContains(arg.message(), partial_error_message);
}

absl::Status CreateDefaultBasicBatchScheduler(
    const BasicBatchScheduler<SavedModelBatchingTask>::Options &options,
    std::function<void(std::unique_ptr<Batch<SavedModelBatchingTask>>)>
        process_batch_callback,
    std::unique_ptr<BatchScheduler<SavedModelBatchingTask>> *batch_scheduler) {
  std::unique_ptr<BasicBatchScheduler<SavedModelBatchingTask>>
      basic_batch_scheduler;
  TF_RETURN_IF_ERROR(BasicBatchScheduler<SavedModelBatchingTask>::Create(
      options, process_batch_callback, &basic_batch_scheduler));
  *batch_scheduler = std::move(basic_batch_scheduler);
  return absl::Status();
}

class SavedModelWithBatchingTest : public ::testing::Test {
 protected:
  SavedModelWithBatchingTest() = default;

  std::unique_ptr<test_util::MockSavedModel> InitializeMockSavedModel() {
    auto wrapped_saved_model = absl::make_unique<test_util::MockSavedModel>();
    wrapped_saved_model_ = wrapped_saved_model.get();
    ON_CALL(*wrapped_saved_model_, GetFunctionMetadata(_))
        .WillByDefault(Return(tfrt::FunctionMetadata(&signature)));
    return wrapped_saved_model;
  }

  void Initialize(
      const BasicBatchScheduler<SavedModelBatchingTask>::Options
          &scheduler_options =
              BasicBatchScheduler<SavedModelBatchingTask>::Options(),
      const SavedModelBatchingOptions &options = SavedModelBatchingOptions()) {
    std::unique_ptr<test_util::MockSavedModel> wrapped_saved_model =
        InitializeMockSavedModel();
    auto scheduler_creator =
        absl::bind_front(&CreateDefaultBasicBatchScheduler, scheduler_options);

    std::vector<FuncNameWithBatchingSchedulerCreator> creators = {
        {kFunctionOne, scheduler_creator}, {kFunctionTwo, scheduler_creator}};
    TF_CHECK_OK(CreateSavedModelWithBatching(options, creators,
                                             std::move(wrapped_saved_model),
                                             &saved_model_with_batching_));
  }

  Tensor MakeTensor(const std::vector<float> &tensor_vec,
                    const TensorShape &shape) {
    return test::AsTensor<float>(tensor_vec, shape);
  }

  std::vector<Tensor> MakeTensors(
      const std::vector<std::pair<std::vector<float>, TensorShape>> &tensors) {
    std::vector<Tensor> inputs;
    inputs.reserve(tensors.size());
    for (const auto &entry : tensors) {
      inputs.push_back(MakeTensor(entry.first, entry.second));
    }
    return inputs;
  }

  std::vector<std::vector<Tensor>> MakeTensorsBatch(
      const std::vector<std::vector<std::pair<std::vector<float>, TensorShape>>>
          &tensor_batch) {
    std::vector<std::vector<Tensor>> result;
    for (const auto &tensors : tensor_batch) {
      result.push_back(MakeTensors(tensors));
    }
    return result;
  }

  std::unique_ptr<tfrt::SavedModel> saved_model_with_batching_;
  test_util::MockSavedModel *wrapped_saved_model_;
};

SavedModelBatchingOptions BuildSavedModelBatchingOptions(
    bool pad_variable_length_inputs, std::vector<int> allowed_batch_sizes) {
  SavedModelBatchingOptions options;
  options.pad_variable_length_inputs = pad_variable_length_inputs;
  options.allowed_batch_sizes = std::move(allowed_batch_sizes);
  return options;
}

// Builds BasicBatchingScheduler options. Only tunnable parameter is
// `max_batch_size`, as we fully use it to control batching behavior.
BasicBatchScheduler<SavedModelBatchingTask>::Options BuildSchedulerOptions(
    int max_batch_size) {
  BasicBatchScheduler<SavedModelBatchingTask>::Options options;
  options.max_batch_size = max_batch_size;
  options.batch_timeout_micros = 1000 * 1000 * 1000;  // 1000s.
  options.num_batch_threads = 1;
  return options;
}

// Expands the `tensor_vec` along 0th dimension by `dim0_size` times.
std::vector<float> ExpandTensor(const std::vector<float> &tensor_vec,
                                int64_t dim0_size) {
  std::vector<float> result;
  for (int i = 0; i < dim0_size; ++i) {
    result.insert(result.end(), tensor_vec.begin(), tensor_vec.end());
  }
  return result;
}

// Tests that creation of SavedModelWithBatching returns an appropriate error if
// the passed underlying SavedMode is NULL.
TEST_F(SavedModelWithBatchingTest, NullWrappedSavedModel) {
  const string error = "must not be null";
  EXPECT_THAT(
      CreateSavedModelWithBatching(SavedModelBatchingOptions(), {}, nullptr,
                                   &saved_model_with_batching_),
      TFStatusIs(error::FAILED_PRECONDITION, error));
}

// Tests that creation of SavedModelWithBatching returns an appropriate error if
// multiple scheduler creators are specified for each function.
TEST_F(SavedModelWithBatchingTest, MultipleBatchSchedulersForOneFunction) {
  std::unique_ptr<test_util::MockSavedModel> wrapped_saved_model =
      InitializeMockSavedModel();
  auto scheduler_creator =
      absl::bind_front(&CreateDefaultBasicBatchScheduler,
                       BasicBatchScheduler<SavedModelBatchingTask>::Options());

  const string error = "multiple batch schedulers";
  std::vector<FuncNameWithBatchingSchedulerCreator> creators = {
      {kFunctionOne, scheduler_creator}, {kFunctionOne, scheduler_creator}};
  EXPECT_THAT(CreateSavedModelWithBatching(
                  SavedModelBatchingOptions(), creators,
                  std::move(wrapped_saved_model), &saved_model_with_batching_),
              TFStatusIs(error::FAILED_PRECONDITION, error));
}

// Tests that creation of SavedModelWithBatching returns an appropriate error if
// a scheduler failed being created.
TEST_F(SavedModelWithBatchingTest, FailedCreatingBatchScheduler) {
  std::unique_ptr<test_util::MockSavedModel> wrapped_saved_model =
      InitializeMockSavedModel();
  auto scheduler_creator =
      [](std::function<void(std::unique_ptr<Batch<SavedModelBatchingTask>>)>
             process_batch_callback,
         std::unique_ptr<BatchScheduler<SavedModelBatchingTask>>
             *batch_scheduler) { return absl::Status(); };

  const string error = "Failed to create batch scheduler";
  std::vector<FuncNameWithBatchingSchedulerCreator> creators = {
      {kFunctionOne, scheduler_creator}};
  EXPECT_THAT(CreateSavedModelWithBatching(
                  SavedModelBatchingOptions(), creators,
                  std::move(wrapped_saved_model), &saved_model_with_batching_),
              TFStatusIs(error::FAILED_PRECONDITION, error));
}

// Tests that when Run() is invoked with a function without a scheduler, it
// delegates to the underlying wrapped SavedModel directly.
TEST_F(SavedModelWithBatchingTest, FunctionNameNotFound) {
  Initialize(BuildSchedulerOptions(/*max_batch_size=*/3));
  std::vector<float> input_tensor_vec1 = {1, 2, 3};
  std::vector<float> input_tensor_vec2 = {2, 3, 4};
  TensorShape input_shape = {1, 3};
  std::vector<Tensor> inputs = MakeTensors(
      {{input_tensor_vec1, input_shape}, {input_tensor_vec2, input_shape}});

  std::vector<float> output_tensor_vec = {2, 2, 3};
  TensorShape output_shape = {1, 3};
  std::vector<Tensor> expected_outputs =
      MakeTensors({{output_tensor_vec, output_shape}});
  std::vector<Tensor> outputs;

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kUnknownFunction, ::testing::An<absl::Span<const Tensor>>(), _))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        outputs->push_back(MakeTensor(output_tensor_vec, output_shape));
        return absl::Status();
      }));
  tfrt::SavedModel::RunOptions run_options;
  // If a corresponding scheduler is found, Run should block forever since
  // maximum batch size isn't reached.
  TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kUnknownFunction,
                                               inputs, &outputs));
  EXPECT_THAT(outputs, ElementsAre(MatchesTensor(&expected_outputs[0])));
}

// Tests Basic batching behavior without any padding.
TEST_F(SavedModelWithBatchingTest, BatchingWithoutPadding) {
  Initialize(BuildSchedulerOptions(/*max_batch_size=*/3));

  std::vector<float> input_tensor1_vec1 = {1, 2, 3};
  std::vector<float> input_tensor1_vec2 = {1, 3, 4};
  std::vector<float> input_tensor2_vec1 = {1, 2, 3, 1, 2, 3};
  std::vector<float> input_tensor2_vec2 = {1, 3, 4, 1, 3, 4};

  TensorShape input1_shape = {1, 3};
  TensorShape input2_shape = {2, 3};
  TensorShape combined_shape = {3, 3};

  auto inputs = MakeTensorsBatch({
      {{input_tensor1_vec1, input1_shape}, {input_tensor1_vec2, input1_shape}},
      {{input_tensor2_vec1, input2_shape}, {input_tensor2_vec2, input2_shape}},
  });

  std::vector<Tensor> combined_inputs =
      MakeTensors({{ExpandTensor(input_tensor1_vec1, 3), combined_shape},
                   {ExpandTensor(input_tensor1_vec2, 3), combined_shape}});

  std::vector<float> output_tensor1_vec = {1, 5, 5, 5};
  std::vector<float> output_tensor2_vec = {1, 5, 5, 5, 1, 5, 5, 5};
  TensorShape output1_shape = {1, 4};
  TensorShape output2_shape = {2, 4};

  auto expected_outputs =
      MakeTensorsBatch({{{output_tensor1_vec, output1_shape}},
                        {{output_tensor2_vec, output2_shape}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        absl::Span<const Tensor> span(inputs);
        EXPECT_THAT(span, ElementsAre(MatchesTensor(&combined_inputs[0]),
                                      MatchesTensor(&combined_inputs[1])));

        // Output is concatenation of `output_tensor1_vec` and
        // `output_tensor2_vec`, in one of the two orders.
        outputs->push_back(MakeTensor(ExpandTensor(output_tensor1_vec, 3),
                                      /*shape=*/{3, 4}));
        return absl::Status();
      }));

  tfrt::SavedModel::RunOptions run_options;
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[0], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[0][0])));
      }));

  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[1], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[1][0])));
      }));
}

// Tests the batching behavior when padding is required both to extend each
// tensor's dimension size and to pad dummy tensors to fit target batch size.
TEST_F(SavedModelWithBatchingTest, BatchingWithPadding) {
  // Need to pad 2 dummy tensors.
  int batch_size = 5;
  Initialize(
      BuildSchedulerOptions(/*max_batch_size=*/3),
      BuildSavedModelBatchingOptions(/*pad_variable_length_inputs=*/true,
                                     /*allowed_batch_sizes=*/{batch_size}));

  std::vector<float> input_tensor1_vec1 = {1, 2, 1, 3};
  std::vector<float> input_tensor1_vec2 = {1, 3, 5, 1, 3, 4};
  std::vector<float> input_tensor2_vec1 = {1, 2, 3};
  std::vector<float> input_tensor2_vec2 = {1, 3, 4, 5};
  // Need to extend 1st dimension.
  TensorShape input1_shape1 = {2, 2};
  TensorShape input1_shape2 = {2, 3};
  TensorShape input2_shape1 = {1, 3};
  TensorShape input2_shape2 = {1, 4};

  auto inputs = MakeTensorsBatch({{{input_tensor1_vec1, input1_shape1},
                                   {input_tensor1_vec2, input1_shape2}},
                                  {{input_tensor2_vec1, input2_shape1},
                                   {input_tensor2_vec2, input2_shape2}}});

  std::vector<float> output_tensor1_vec = {1, 5, 5, 1, 5, 5};
  std::vector<float> output_tensor2_vec = {1, 5, 5};
  TensorShape output1_shape = {2, 3};
  TensorShape output2_shape = {1, 3};

  auto expected_outputs =
      MakeTensorsBatch({{{output_tensor1_vec, output1_shape}},
                        {{output_tensor2_vec, output2_shape}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        // First input tensor is of shape (5, 3)
        EXPECT_EQ(15, inputs[0].NumElements());
        // Second input tensor is of shape (5, 4)
        EXPECT_EQ(20, inputs[1].NumElements());

        // Output is concatenation of `output_tensor1_vec` and
        // `output_tensor2_vec`, in one of the two orders.
        outputs->push_back(MakeTensor(ExpandTensor({1, 5, 5}, batch_size),
                                      /*shape=*/{batch_size, 3}));
        return absl::Status();
      }));

  tfrt::SavedModel::RunOptions run_options;
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[0], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[0][0])));
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[1], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[1][0])));
      }));
}

// Tests that batching tensors with variable length dimension size (except for
// batching dimension) returns an appropriate error when padding is turned off.
TEST_F(SavedModelWithBatchingTest, UnequalShapesWhenPaddingIsTurnedOff) {
  Initialize(BuildSchedulerOptions(/*max_batch_size=*/2));

  std::vector<float> input_tensor1_vec = {1, 2, 3};
  std::vector<float> input_tensor2_vec = {1, 2, 3, 4};

  TensorShape input1_shape = {1, 3};
  TensorShape input2_shape = {1, 4};

  auto inputs = MakeTensorsBatch({{{input_tensor1_vec, input1_shape}},
                                  {{input_tensor2_vec, input2_shape}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(0);

  tfrt::SavedModel::RunOptions run_options;
  const string error = "different shapes other than first dimension";
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        EXPECT_THAT(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                    inputs[0], &outputs),
                    TFStatusIs(error::FAILED_PRECONDITION, error));
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        EXPECT_THAT(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                    inputs[1], &outputs),
                    TFStatusIs(error::FAILED_PRECONDITION, error));
      }));
}

// Tests that processing batch returns an appropriate error if all tasks in the
// batch has a past deadline.
TEST_F(SavedModelWithBatchingTest, AllTasksExceededDeadline) {
  Initialize(BuildSchedulerOptions(/*max_batch_size=*/2));

  std::vector<float> input_tensor1_vec = {1, 2, 3};
  std::vector<float> input_tensor2_vec = {1, 2, 4};

  TensorShape input_shape = {1, 3};

  auto inputs = MakeTensorsBatch(
      {{{input_tensor1_vec, input_shape}}, {{input_tensor2_vec, input_shape}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(0);

  tfrt::SavedModel::RunOptions run_options;
  run_options.deadline = absl::ToChronoTime(absl::Now());
  const string error = "timeout exceeded";
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        EXPECT_THAT(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                    inputs[0], &outputs),
                    TFStatusIs(error::RESOURCE_EXHAUSTED, error));
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        EXPECT_THAT(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                    inputs[1], &outputs),
                    TFStatusIs(error::RESOURCE_EXHAUSTED, error));
      }));
}

// Tests that distinct functions should be batched independently.
TEST_F(SavedModelWithBatchingTest, MultipleFunctions) {
  Initialize(BuildSchedulerOptions(/*max_batch_size=*/3));

  std::vector<float> input_tensor1_vec = {1, 3, 4};
  std::vector<float> input_tensor2_vec = {2, 4, 5};
  std::vector<float> input_tensor3_vec = {1, 3, 4, 1, 3, 4};
  std::vector<float> input_tensor4_vec = {2, 4, 5, 2, 4, 5};

  TensorShape input_shape1 = {1, 3};
  TensorShape input_shape2 = {2, 3};
  TensorShape combined_shape = {3, 3};

  std::vector<std::vector<Tensor>> inputs =
      MakeTensorsBatch({{{input_tensor1_vec, input_shape1}},
                        {{input_tensor2_vec, input_shape1}},
                        {{input_tensor3_vec, input_shape2}},
                        {{input_tensor4_vec, input_shape2}}});

  std::vector<Tensor> combined_inputs1 =
      MakeTensors({{ExpandTensor(input_tensor1_vec, 3), combined_shape}});
  std::vector<Tensor> combined_inputs2 =
      MakeTensors({{ExpandTensor(input_tensor2_vec, 3), combined_shape}});

  std::vector<float> output_tensor1_vec = {1, 5, 5, 5};
  std::vector<float> output_tensor2_vec = {1, 6, 6, 6};
  std::vector<float> output_tensor3_vec = {1, 5, 5, 5, 1, 5, 5, 5};
  std::vector<float> output_tensor4_vec = {1, 6, 6, 6, 1, 6, 6, 6};

  TensorShape output1_shape = {1, 4};
  TensorShape output2_shape = {2, 4};

  std::vector<std::vector<Tensor>> expected_outputs = MakeTensorsBatch({
      {{output_tensor1_vec, output1_shape}},
      {{output_tensor2_vec, output1_shape}},
      {{output_tensor3_vec, output2_shape}},
      {{output_tensor4_vec, output2_shape}},
  });

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        absl::Span<const Tensor> span(inputs);
        EXPECT_THAT(span, ElementsAre(MatchesTensor(&combined_inputs1[0])));

        // Output is concatenation of `output_tensor1_vec` and
        // `output_tensor2_vec`, in one of the two orders.
        outputs->push_back(
            MakeTensor(ExpandTensor(output_tensor1_vec, 3), {3, 4}));
        return absl::Status();
      }));

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionTwo, ::testing::An<absl::Span<const Tensor>>(), _))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        absl::Span<const Tensor> span(inputs);
        EXPECT_THAT(span, ElementsAre(MatchesTensor(&combined_inputs2[0])));

        // Output is concatenation of `output_tensor3_vec` and
        // `output_tensor4_vec`, in one of the two orders.
        outputs->push_back(
            MakeTensor(ExpandTensor(output_tensor2_vec, 3), {3, 4}));
        return absl::Status();
      }));

  std::vector<std::unique_ptr<Thread>> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(std::unique_ptr<Thread>(Env::Default()->StartThread(
        ThreadOptions(), absl::StrCat("request_thread_", i),
        [this, i, &inputs, &expected_outputs] {
          std::vector<Tensor> outputs;
          TF_ASSERT_OK(saved_model_with_batching_->Run(
              tfrt::SavedModel::RunOptions(),
              i == 0 || i == 2 ? kFunctionOne : kFunctionTwo, inputs[i],
              &outputs));
          EXPECT_THAT(outputs,
                      ElementsAre(MatchesTensor(&expected_outputs[i][0])));
        })));
  }
}

// Tests that when a large batch needs to be splitted, tensors are splitted and
// partial outputs are eventually merged appropriately.
TEST_F(SavedModelWithBatchingTest, SplitInputBasic) {
  const int batch_size = 3;
  BasicBatchScheduler<SavedModelBatchingTask>::Options options =
      BuildSchedulerOptions(6);
  options.enable_large_batch_splitting = true;
  options.max_execution_batch_size = batch_size;
  options.split_input_task_func = SplitSavedModelInputTask;
  Initialize(options, BuildSavedModelBatchingOptions(
                          /*pad_variable_length_inputs=*/true,
                          /*allowed_batch_sizes=*/{batch_size}));

  std::vector<float> input_tensor1_vec1 = {1, 2, 3, 4, 5, 6};
  std::vector<float> input_tensor1_vec2 = {1, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> input_tensor2_vec1 = {1, 2, 1, 3};
  std::vector<float> input_tensor2_vec2 = {1, 3, 5, 1, 3, 4, 5, 6};

  TensorShape input1_shape1 = {2, 3};
  TensorShape input1_shape2 = {2, 4};
  TensorShape input2_shape1 = {4, 1};
  TensorShape input2_shape2 = {4, 2};

  auto inputs = MakeTensorsBatch({{{input_tensor1_vec1, input1_shape1},
                                   {input_tensor1_vec2, input1_shape2}},
                                  {{input_tensor2_vec1, input2_shape1},
                                   {input_tensor2_vec2, input2_shape2}}});

  std::vector<float> output_tensor1_vec = {1, 5, 5, 1, 5, 5};
  std::vector<float> output_tensor2_vec = {1, 5, 5, 1, 5, 5, 1, 5, 5, 1, 5, 5};
  TensorShape output1_shape = {2, 3};
  TensorShape output2_shape = {4, 3};

  auto expected_outputs =
      MakeTensorsBatch({{{output_tensor1_vec, output1_shape}},
                        {{output_tensor2_vec, output2_shape}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(2)
      .WillRepeatedly(Invoke(
          [&](const tfrt::SavedModel::RunOptions &run_options,
              absl::string_view func_name, absl::Span<const Tensor> inputs,
              std::vector<Tensor> *outputs) {
            // Second input (batch size 4) should be split into two tasks, one
            // with batch size 1 and batches with the first input, one with
            // batch size 3 which should stay itself. We only verify the first
            // dimension is 3 because we don't know which batch comes first.
            EXPECT_EQ(3, inputs[0].dim_size(0));
            // Second input tensor is of shape (3, 2)
            EXPECT_EQ(3, inputs[1].dim_size(0));

            // Output is concatenation of `output_tensor1_vec` and
            // `output_tensor2_vec`, in one of the two orders.
            outputs->push_back(MakeTensor(ExpandTensor({1, 5, 5}, batch_size),
                                          /*shape=*/{batch_size, 3}));
            return absl::Status();
          }));

  tfrt::SavedModel::RunOptions run_options;
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[0], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[0][0])));
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                     inputs[1], &outputs));
        EXPECT_THAT(outputs,
                    ElementsAre(MatchesTensor(&expected_outputs[1][0])));
      }));
}

TEST_F(SavedModelWithBatchingTest, PartialTaskFails) {
  const int batch_size = 3;
  BasicBatchScheduler<SavedModelBatchingTask>::Options options =
      BuildSchedulerOptions(6);
  options.enable_large_batch_splitting = true;
  options.max_execution_batch_size = batch_size;
  options.split_input_task_func = SplitSavedModelInputTask;
  Initialize(options, BuildSavedModelBatchingOptions(
                          /*pad_variable_length_inputs=*/true,
                          /*allowed_batch_sizes=*/{batch_size}));

  std::vector<float> input_tensor1_vec1 = {1, 2, 3, 4, 5, 6};
  std::vector<float> input_tensor1_vec2 = {1, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> input_tensor2_vec1 = {1, 2, 1, 3};
  std::vector<float> input_tensor2_vec2 = {1, 3, 5, 1, 3, 4, 5, 6};

  TensorShape input1_shape1 = {2, 3};
  TensorShape input1_shape2 = {2, 4};
  TensorShape input2_shape1 = {4, 1};
  TensorShape input2_shape2 = {4, 2};

  auto inputs = MakeTensorsBatch({{{input_tensor1_vec1, input1_shape1},
                                   {input_tensor1_vec2, input1_shape2}},
                                  {{input_tensor2_vec1, input2_shape1},
                                   {input_tensor2_vec2, input2_shape2}}});

  EXPECT_CALL(
      *wrapped_saved_model_,
      Run(_, kFunctionOne, ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(2)
      // Fail One of the two partial tasks.
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        return errors::Internal("Error");
      }))
      .WillOnce(Invoke([&](const tfrt::SavedModel::RunOptions &run_options,
                           absl::string_view func_name,
                           absl::Span<const Tensor> inputs,
                           std::vector<Tensor> *outputs) {
        // Output is concatenation of `output_tensor1_vec` and
        // `output_tensor2_vec`, in one of the two orders.
        outputs->push_back(MakeTensor(ExpandTensor({1, 5, 5}, batch_size),
                                      /*shape=*/{batch_size, 3}));
        return absl::Status();
      }));

  tfrt::SavedModel::RunOptions run_options;
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        // First input may or may not succeed, because it is only in one of the
        // two batches and only one batch fails.
        const absl::Status ignore_result = saved_model_with_batching_->Run(
            run_options, kFunctionOne, inputs[0], &outputs);
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        // Second input must fail, since it is splitted into two partial tasks
        // which are in different batches. Thus failure of any one of the two
        // batches will fail this the second input.
        EXPECT_THAT(saved_model_with_batching_->Run(run_options, kFunctionOne,
                                                    inputs[1], &outputs),
                    TFStatusIs(error::INTERNAL, "Error"));
      }));
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
