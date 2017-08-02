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
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
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
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

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
    RunMetadata run_metadata;
    return Run(RunOptions(), inputs, output_tensor_names, target_node_names,
               outputs, &run_metadata);
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
    latest_batch_size_ = inputs[0].second.shape().dim_size(0);
    return wrapped_->Run(run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
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
  tensorflow::RunOptions run_options;
  const string export_dir = test_util::TensorflowTestSrcDirPath(
      "cc/saved_model/testdata/half_plus_two/00000123");
  SavedModelBundle bundle;
  TF_CHECK_OK(LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle));
  return std::move(bundle.session);
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

// Creates a SignatureDef from a TensorSignature.
SignatureDef CreateSignatureDef(const TensorSignature& tensor_signature) {
  SignatureDef signature_def;
  for (const string& input_tensor : tensor_signature.input_tensors) {
    TensorInfo input;
    input.set_name(input_tensor);
    (*signature_def.mutable_inputs())[input_tensor] = input;
  }
  for (const string& output_tensor : tensor_signature.output_tensors) {
    TensorInfo output;
    output.set_name(output_tensor);
    (*signature_def.mutable_outputs())[output_tensor] = output;
  }
  return signature_def;
}

TEST(BatchingSessionTest, TensorSignatureFromSignatureDef) {
  const SignatureDef signature_def =
      CreateSignatureDef({{"x0", "x1"}, {"y0", "y1"}});
  const TensorSignature tensor_signature =
      TensorSignatureFromSignatureDef(signature_def);
  EXPECT_THAT(tensor_signature.input_tensors, UnorderedElementsAre("x0", "x1"));
  EXPECT_THAT(tensor_signature.output_tensors,
              UnorderedElementsAre("y0", "y1"));
}

TEST(BatchingSessionTest, TensorSignatureFromSignatureDefs) {
  const SignatureDef signature_def_0 =
      CreateSignatureDef({{"x0", "x1"}, {"y0", "y1"}});
  const SignatureDef signature_def_1 =
      CreateSignatureDef({{"x1", "x2"}, {"y1", "y3"}});
  const TensorSignature tensor_signature =
      TensorSignatureFromSignatureDefs({signature_def_0, signature_def_1});
  EXPECT_THAT(tensor_signature.input_tensors,
              UnorderedElementsAre("x0", "x1", "x2"));
  EXPECT_THAT(tensor_signature.output_tensors,
              UnorderedElementsAre("y0", "y1", "y3"));
}

TEST(BatchingSessionTest, Basic) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
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
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));
  TestSingleRequest(100.0f, 42.0f, batching_session.get());
}

TEST(BatchingSessionTest, RequestThatDoesntMatchSignatureGetsRunAnyway) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  // Set the batching parameters s.t. if the request is batched the test will
  // timeout.
  schedule_options.max_batch_size = 100;
  schedule_options.batch_timeout_micros = INT_MAX;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x2"}, {"y3"}},
      CreateHalfPlusTwoSession(), &batching_session));
  // Issue a request using x/y, which doesn't match the x2/y3 signature.
  TestSingleRequest(100.0f, 42.0f, batching_session.get());
}

TEST(BatchingSessionTest, RequestWithIncompatibleInputTensorSizes) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;

  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options,
      {{"input_0", "input_1"}, {"output"}}, CreateHalfPlusTwoSession(),
      &batching_session));

  ExpectError(
      "Batching session Run() input tensors must have equal 0th-dimension size",
      {{"input_0", test::AsTensor<int>({3}, {1})},
       {"input_1", test::AsTensor<int>({5, 7}, {2})}},
      {"output"}, batching_session.get());
}

TEST(BatchingSessionTest, AllowedBatchSizesNoPaddingNeeded) {
  // Arrange to capture the batch size.
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
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
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {1, 3, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      std::move(batch_size_capturing_session), &batching_session));
  TestSingleRequest(100.0f, 42.0f, batching_session.get());

  // It should pad the batch size from 2 to 3.
  EXPECT_EQ(3, batch_size_capturing_session_raw->latest_batch_size());
}

TEST(BatchingSessionTest, UnsortedAllowedBatchSizesRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {4, 2};  // Not sorted.
  std::unique_ptr<Session> batching_session;
  EXPECT_FALSE(CreateBasicBatchingSession(
                   schedule_options, batching_session_options, {{"x"}, {"y"}},
                   CreateHalfPlusTwoSession(), &batching_session)
                   .ok());
}

TEST(BatchingSessionTest,
     FinalAllowedBatchSizeDifferingFromMaxBatchSizeRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 8};  // Final entry != 4.
  std::unique_ptr<Session> batching_session;
  EXPECT_FALSE(CreateBasicBatchingSession(
                   schedule_options, batching_session_options, {{"x"}, {"y"}},
                   CreateHalfPlusTwoSession(), &batching_session)
                   .ok());
}

TEST(BatchingSessionTest, DifferentOrderForInputAndOutputTensors) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 6;  // fits three 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  BatchingSessionOptions batching_session_options;
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x", "x2"}, {"y", "y3"}},
      CreateHalfPlusTwoSession(), &batching_session));

  const Tensor input0 = test::AsTensor<float>({8.0f, 6.0f}, {2});
  const Tensor expected_output0 = test::AsTensor<float>({6.0f, 5.0f}, {2});
  const Tensor input1 = test::AsTensor<float>({100.0f, 42.0f}, {2});
  const Tensor expected_output1 = test::AsTensor<float>({53.0f, 24.0f}, {2});

  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(batching_session->Run({{"x", input0}, {"x2", input1}},
                                           {"y", "y3"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(2, outputs.size());
        test::ExpectTensorEqual<float>(expected_output0, outputs[0]);
        test::ExpectTensorEqual<float>(expected_output1, outputs[1]);
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(batching_session->Run({{"x2", input1}, {"x", input0}},
                                           {"y3", "y"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(2, outputs.size());
        test::ExpectTensorEqual<float>(expected_output1, outputs[0]);
        test::ExpectTensorEqual<float>(expected_output0, outputs[1]);
      }));
  std::unique_ptr<Thread> third_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "third_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(batching_session->Run({{"x2", input1}, {"x", input0}},
                                           {"y", "y3"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(2, outputs.size());
        test::ExpectTensorEqual<float>(expected_output0, outputs[0]);
        test::ExpectTensorEqual<float>(expected_output1, outputs[1]);
      }));
}

TEST(BatchingSessionTest, MultipleSignatures) {
  std::vector<BatchScheduler<BatchingSessionTask>*> schedulers;
  auto create_scheduler = [&schedulers](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* scheduler) {
    BasicBatchScheduler<BatchingSessionTask>::Options options;
    options.max_batch_size = 4;                      // fits two 2-unit tasks
    options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
    options.num_batch_threads = 1;
    std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>> basic_scheduler;
    TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
        options, process_batch_callback, &basic_scheduler));
    schedulers.push_back(basic_scheduler.get());
    *scheduler = std::move(basic_scheduler);
    return Status::OK();
  };
  BatchingSessionOptions batching_session_options;
  std::unique_ptr<Session> batching_session;
  TF_CHECK_OK(CreateBatchingSession(
      batching_session_options, {{{{"x"}, {"y"}}, create_scheduler},
                                 {{{"x2"}, {"y3"}}, create_scheduler}},
      CreateHalfPlusTwoSession(), &batching_session));
  ASSERT_EQ(2, schedulers.size());

  // Create lambdas for 2-unit inference requests to each signature.
  auto run_signature0_request = [&batching_session] {
    Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    Tensor expected_output = test::AsTensor<float>({52.0f, 23.0f}, {2});
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(batching_session->Run({{"x", input}}, {"y"} /* outputs */,
                                       {} /* target nodes */, &outputs));
    ASSERT_EQ(1, outputs.size());
    test::ExpectTensorEqual<float>(expected_output, outputs[0]);
  };
  auto run_signature1_request = [&batching_session] {
    Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    Tensor expected_output = test::AsTensor<float>({53.0f, 24.0f}, {2});
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(batching_session->Run({{"x2", input}}, {"y3"} /* outputs */,
                                       {} /* target nodes */, &outputs));
    ASSERT_EQ(1, outputs.size());
    test::ExpectTensorEqual<float>(expected_output, outputs[0]);
  };

  // Enqueue one request for each signature. Both should block because neither
  // batching queue will be full yet.
  std::unique_ptr<Thread> signature0_thread(Env::Default()->StartThread(
      ThreadOptions(), "signature0_thread", [&] { run_signature0_request(); }));
  std::unique_ptr<Thread> signature1_thread(Env::Default()->StartThread(
      ThreadOptions(), "signature1_thread", [&] { run_signature1_request(); }));
  while (schedulers[0]->NumEnqueuedTasks() != 1 &&
         schedulers[1]->NumEnqueuedTasks() != 1) {
    Env::Default()->SleepForMicroseconds(100);
  }

  // Enqueue a second request for each signature. This should fill both queues
  // and unblock all the processing.
  run_signature0_request();
  EXPECT_EQ(0, schedulers[0]->NumEnqueuedTasks());
  run_signature1_request();
  EXPECT_EQ(0, schedulers[1]->NumEnqueuedTasks());
}

TEST(BatchingSessionTest, EnqueuedLongerThanTimeout) {
  BatchScheduler<BatchingSessionTask>* scheduler = nullptr;
  auto create_scheduler = [&scheduler](
      std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
          process_batch_callback,
      std::unique_ptr<BatchScheduler<BatchingSessionTask>>* new_scheduler) {
    BasicBatchScheduler<BatchingSessionTask>::Options options;
    options.max_batch_size = 4;                      // fits two 2-unit tasks
    options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
    options.num_batch_threads = 1;
    std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>> basic_scheduler;
    TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
        options, process_batch_callback, &basic_scheduler));
    scheduler = basic_scheduler.get();
    *new_scheduler = std::move(basic_scheduler);
    return Status::OK();
  };
  BatchingSessionOptions batching_session_options;
  std::unique_ptr<Session> batching_session;
  TF_CHECK_OK(CreateBatchingSession(
      batching_session_options, {{{{"x"}, {"y"}}, create_scheduler}},
      CreateHalfPlusTwoSession(), &batching_session));
  ASSERT_FALSE(scheduler == nullptr);

  // Enqueue a request with a timeout specified via RunOptions.
  Notification request_returned;
  auto issue_request = [&batching_session, &request_returned] {
    Tensor input = test::AsTensor<float>({100.0f, 42.0f}, {2});
    RunOptions run_options;
    run_options.set_timeout_in_ms(1);
    std::vector<Tensor> outputs;
    RunMetadata run_metadata;
    const Status status =
        batching_session->Run(run_options, {{"x", input}}, {"y"} /* outputs */,
                              {} /* target nodes */, &outputs, &run_metadata);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(error::RESOURCE_EXHAUSTED, status.code());
    EXPECT_THAT(
        status.error_message(),
        HasSubstr("Run() timeout exceeded while waiting in batching queue"));
    request_returned.Notify();
  };
  std::unique_ptr<Thread> request_thread(Env::Default()->StartThread(
      ThreadOptions(), "request_thread", [&] { issue_request(); }));
  while (scheduler->NumEnqueuedTasks() != 1) {
    Env::Default()->SleepForMicroseconds(100);
  }
  // Sleep for longer than the request's timeout, so that when it does finally
  // get dequeued for batch processing it has already exceeded its timeout.
  Env::Default()->SleepForMicroseconds(10 * 1000);
  // Tear down the batcher, so that it schedules the pending batch.
  batching_session = nullptr;
  request_returned.WaitForNotification();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
