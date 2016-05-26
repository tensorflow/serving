/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/batching/batching_session.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/batching/batch_scheduler_retrier.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {

// A wrapper around a Session that captures the batch size.
class BatchSizeCapturingSession : public ServingSession {
 public:
  explicit BatchSizeCapturingSession(std::unique_ptr<Session> wrapped)
      : wrapped_(std::move(wrapped)) {}
  ~BatchSizeCapturingSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override {
    latest_batch_size_ = inputs[0].second.shape().dim_size(0);
    return wrapped_->Run(inputs, output_tensor_names, target_node_names,
                         outputs);
  }

  int latest_batch_size() const { return latest_batch_size_; }

 private:
  std::unique_ptr<Session> wrapped_;

  // The size of the batch most recently submitted to Run().
  int latest_batch_size_ = -1;

  TF_DISALLOW_COPY_AND_ASSIGN(BatchSizeCapturingSession);
};

// Creates a (non-batching) session with the half-plus-two model loaded.
std::unique_ptr<Session> CreateHalfPlusTwoSession() {
  tensorflow::SessionOptions session_options;
  const string export_dir = test_util::TestSrcDirPath(
      "session_bundle/example/half_plus_two/00000123");
  SessionBundle session_bundle;
  TF_CHECK_OK(
      LoadSessionBundleFromPath(session_options, export_dir, &session_bundle));
  return std::move(session_bundle.session);
}

// Test that a session handles a single request for the half-plus-two model
// properly. The request has two input floats (size=2 for batching purposes).
void TestSingleRequest(float input_0, float input_1, Session* session) {
  Tensor input = test::AsTensor<float>({input_0, input_1}, {2});
  // Half plus two: each output should be input / 2 + 2.
  Tensor expected_output =
      test::AsTensor<float>({input_0 / 2 + 2, input_1 / 2 + 2}, {2});

  const std::vector<std::pair<string, Tensor>> inputs = {{"x", input}};
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(inputs, {"y"}, {} /* target nodes */, &outputs));
  ASSERT_EQ(1, outputs.size());
  test::ExpectTensorEqual<float>(expected_output, outputs[0]);
}

// Invoke Run() with the supplied arguments, and expect a particular error.
void ExpectError(const string& error_message,
                 const std::vector<std::pair<string, Tensor>>& inputs,
                 const std::vector<string>& output_tensor_names,
                 Session* session) {
  std::vector<Tensor> outputs;
  Status status = session->Run(inputs, output_tensor_names,
                               {} /* target nodes */, &outputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(error_message, status.error_message());
}

TEST(BatchingSessionTest, Basic) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      CreateHalfPlusTwoSession(), &batching_session));

  // Asynchronously send two requests whose total size is 4. The two requests in
  // conjunction should trigger a batch to be processed.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request_thread", [&batching_session] {
        TestSingleRequest(100.0f, 42.0f, batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&batching_session] {
        TestSingleRequest(71.5f, 18.3f, batching_session.get());
      }));
}

TEST(BatchingSessionTest, SingletonBatch) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      CreateHalfPlusTwoSession(), &batching_session));
  TestSingleRequest(100.0f, 42.0f, batching_session.get());
}

TEST(BatchingSessionTest, RequestWithIncompatibleInputTensorSizes) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      CreateHalfPlusTwoSession(), &batching_session));

  ExpectError(
      "Batching session Run() input tensors must have equal 0th-dimension size",
      {{"input_0", test::AsTensor<int>({3}, {1})},
       {"input_1", test::AsTensor<int>({5, 7}, {2})}},
      {"output"}, batching_session.get());
}

TEST(BatchingSessionTest, RequestsWithDifferentInputTensors) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;  // fits two 1-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      CreateHalfPlusTwoSession(), &batching_session));

  std::unique_ptr<Thread> thread_1(Env::Default()->StartThread(
      ThreadOptions(), "thread_1", [&batching_session] {
        ExpectError(
            "Batching session Run() calls must supply the same input tensors",
            {{"input", test::AsTensor<int>({3}, {1})}}, {"output"},
            batching_session.get());
      }));
  std::unique_ptr<Thread> thread_2(Env::Default()->StartThread(
      ThreadOptions(), "thread_2", [&batching_session] {
        ExpectError(
            "Batching session Run() calls must supply the same input tensors",
            {{"input", test::AsTensor<int>({3}, {1})},
             {"another_input", test::AsTensor<int>({3}, {1})}},
            {"output"}, batching_session.get());
      }));
}

TEST(BatchingSessionTest, RequestsWithDifferentOutputTensors) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;  // fits two 1-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      CreateHalfPlusTwoSession(), &batching_session));

  std::unique_ptr<Thread> thread_1(Env::Default()->StartThread(
      ThreadOptions(), "thread_1", [&batching_session] {
        ExpectError(
            "Batching session Run() calls must supply the same output tensors",
            {{"input", test::AsTensor<int>({3}, {1})}}, {"output"},
            batching_session.get());
      }));
  std::unique_ptr<Thread> thread_2(Env::Default()->StartThread(
      ThreadOptions(), "thread_2", [&batching_session] {
        ExpectError(
            "Batching session Run() calls must supply the same output tensors",
            {{"input", test::AsTensor<int>({3}, {1})}},
            {"output", "another_output"}, batching_session.get());
      }));
}

TEST(BatchingSessionTest, AllowedBatchSizes_NoPaddingNeeded) {
  // Arrange to capture the batch size.
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      std::move(batch_size_capturing_session), &batching_session));
  TestSingleRequest(100.0f, 42.0f, batching_session.get());

  // It should not add any padding, i.e. leave the batch size at 2.
  EXPECT_EQ(2, batch_size_capturing_session_raw->latest_batch_size());
}

TEST(BatchingSessionTest, AllowedBatchSizesRequirePadding) {
  // Arrange to capture the batch size.
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {1, 3, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateRetryingBasicBatchingSession(
      schedule_options, retry_options, batching_session_options,
      std::move(batch_size_capturing_session), &batching_session));
  TestSingleRequest(100.0f, 42.0f, batching_session.get());

  // It should pad the batch size from 2 to 3.
  EXPECT_EQ(3, batch_size_capturing_session_raw->latest_batch_size());
}

TEST(BatchingSessionTest, UnsortedAllowedBatchSizesRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {4, 2};  // Not sorted.
  std::unique_ptr<Session> batching_session;
  EXPECT_FALSE(CreateRetryingBasicBatchingSession(
                   schedule_options, retry_options, batching_session_options,
                   CreateHalfPlusTwoSession(), &batching_session)
                   .ok());
}

TEST(BatchingSessionTest,
     FinalAllowedBatchSizeDifferingFromMaxBatchSizeRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  BatchSchedulerRetrier<BatchingSessionTask>::Options retry_options;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 8};  // Final entry != 4.
  std::unique_ptr<Session> batching_session;
  EXPECT_FALSE(CreateRetryingBasicBatchingSession(
                   schedule_options, retry_options, batching_session_options,
                   CreateHalfPlusTwoSession(), &batching_session)
                   .ok());
}

}  // namespace serving
}  // namespace tensorflow
