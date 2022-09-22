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

#include <memory>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
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
    return Run(run_options, inputs, output_tensor_names, target_node_names,
               outputs, run_metadata, thread::ThreadPoolOptions());
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override
      TF_LOCKS_EXCLUDED(latest_batch_size_mu_) {
    {
      mutex_lock l(latest_batch_size_mu_);
      latest_batch_size_ = inputs[0].second.shape().dim_size(0);
    }
    Status status = wrapped_->Run(run_options, inputs, output_tensor_names,
                                  target_node_names, outputs, run_metadata,
                                  thread_pool_options);
    *(run_metadata->mutable_cost_graph()) = cost_graph_;
    return status;
  }

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return wrapped_->ListDevices(response);
  }

  int latest_batch_size() const TF_LOCKS_EXCLUDED(latest_batch_size_mu_) {
    mutex_lock l(latest_batch_size_mu_);
    return latest_batch_size_;
  }

  CostGraphDef* mutable_cost_graph() { return &cost_graph_; }

 private:
  std::unique_ptr<Session> wrapped_;

  mutable mutex latest_batch_size_mu_;
  // The size of the batch most recently submitted to Run().
  int latest_batch_size_ TF_GUARDED_BY(latest_batch_size_mu_) = -1;

  // Cost graph associated with the latest call to Run().
  CostGraphDef cost_graph_;

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

std::unique_ptr<Session> CreateMatrixHalfPlusTwoSession() {
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;
  const string export_dir =
      test_util::TestSrcDirPath("batching/testdata/matrix_half_plus_two/1");
  SavedModelBundle bundle;
  TF_CHECK_OK(LoadSavedModel(session_options, run_options, export_dir,
                             {kSavedModelTagServe}, &bundle));
  return std::move(bundle.session);
}

void TestRequest(const std::vector<float>& x_values, TensorShape x_shape,
                 const std::vector<float>& y_values, TensorShape y_shape,
                 Session* session,
                 test_util::CountingThreadPool* inter_op_threadpool = nullptr,
                 test_util::CountingThreadPool* intra_op_threadpool = nullptr) {
  Tensor input = test::AsTensor<float>(x_values, x_shape);
  Tensor expected_output = test::AsTensor<float>(y_values, y_shape);

  RunMetadata run_metadata;
  thread::ThreadPoolOptions thread_pool_options;
  thread_pool_options.inter_op_threadpool = inter_op_threadpool;
  thread_pool_options.intra_op_threadpool = intra_op_threadpool;
  std::vector<Tensor> output;
  TF_ASSERT_OK(session->Run(RunOptions(), {{"x", input}}, {"y"},
                            {} /* target nodes */, &output, &run_metadata,
                            thread_pool_options));
  ASSERT_EQ(1, output.size());
  test::ExpectTensorEqual<float>(expected_output, output[0]);

  // The intra_op_threadpool doesn't have anything scheduled.
  if (inter_op_threadpool != nullptr) {
    ASSERT_GE(inter_op_threadpool->NumScheduled(), 1);
  }
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

int GetPercentileTotal(string label) {
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics(options);
  int total_samples = 0;
  const auto& point_set_map = collected_metrics->point_set_map;
  if (point_set_map.find(label) == point_set_map.end()) return 0;
  const monitoring::PointSet& lps = *point_set_map.at(label);
  for (int i = 0; i < lps.points.size(); ++i) {
    total_samples += lps.points[i]->histogram_value.sum();
  }
  return static_cast<int>(total_samples);
}

bool CheckDescriptor(string label, const string& description,
                     const std::vector<string>& labels) {
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics(options);
  const auto& metric_descriptor_map = collected_metrics->metric_descriptor_map;
  if (metric_descriptor_map.find(label) == metric_descriptor_map.end()) {
    return false;
  }
  const monitoring::MetricDescriptor& desc = *metric_descriptor_map.at(label);
  if (desc.description != description) return false;
  if (labels.size() != desc.label_names.size()) return false;
  for (int i = 0; i < labels.size(); ++i) {
    if (labels[i] != desc.label_names[i]) return false;
  }
  return true;
}

TEST(BatchingSessionSignatureTest, TensorSignatureFromSignatureDef) {
  const SignatureDef signature_def =
      CreateSignatureDef({{"x0", "x1"}, {"y0", "y1"}});
  const TensorSignature tensor_signature =
      TensorSignatureFromSignatureDef(signature_def);
  EXPECT_THAT(tensor_signature.input_tensors, UnorderedElementsAre("x0", "x1"));
  EXPECT_THAT(tensor_signature.output_tensors,
              UnorderedElementsAre("y0", "y1"));
}

TEST(BatchingSessionSignatureTest, TensorSignatureFromSignatureDefs) {
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

class BatchingSessionTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
 public:
  BatchingSessionTest() {}

  bool enable_large_batch_splitting() const { return std::get<0>(GetParam()); }

  bool enable_lazy_split() const { return std::get<1>(GetParam()); }

  std::function<
      Status(std::unique_ptr<BatchingSessionTask>* input_task,
             int first_output_task_size, int max_batch_size,
             std::vector<std::unique_ptr<BatchingSessionTask>>* output_tasks)>
  get_split_input_task_func() const {
    if (enable_large_batch_splitting()) {
      return SplitInputTask;
    }
    return nullptr;
  }

  // If 'enable_large_batch_splitting' is true, annotate `input_options` with
  // parameters for splitting large batches.
  BasicBatchScheduler<BatchingSessionTask>::Options annotate_options(
      const BasicBatchScheduler<BatchingSessionTask>::Options input_options) {
    BasicBatchScheduler<BatchingSessionTask>::Options output_options =
        input_options;
    output_options.enable_large_batch_splitting =
        enable_large_batch_splitting();
    output_options.enable_lazy_split = enable_lazy_split();
    if (enable_large_batch_splitting()) {
      output_options.split_input_task_func = get_split_input_task_func();
      // Bump up the max batch size, and set execution batch size to the max
      // size we actually want -- this will allow us to exercise large batch
      // splits (they trigger when execution_batch_size < max_batch_size).
      output_options.max_execution_batch_size = input_options.max_batch_size;
      output_options.max_batch_size = input_options.max_batch_size * 2;
    }
    return output_options;
  }
};

TEST_P(BatchingSessionTest, Basic) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);

  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));

  // Asynchronously send two requests whose total size is 4. The two requests in
  // conjunction should trigger a batch to be processed.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request_thread", [&batching_session] {
        TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
                    batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&batching_session] {
        TestRequest({71.5f, 18.3f}, {2}, {37.75f, 11.15f}, {2},
                    batching_session.get());
      }));
}

TEST_P(BatchingSessionTest, BatchingWithPadding) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;
  schedule_options.batch_timeout_micros = 1e6;
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  batching_session_options.pad_variable_length_inputs = true;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateMatrixHalfPlusTwoSession(), &batching_session));
  // two requests form a batch and first input gets padded with zeros to match
  // [1, 3, 3] shape that is accepted by the model.
  // if padding doesn't work, test will fail.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request", [&batching_session] {
        TestRequest({1, 2, 3, 4}, {1, 2, 2},
                    {2.5, 3, 2.5, 3.5, 4, 2.5, 2.5, 2.5, 2.5}, {1, 3, 3},
                    batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request", [&batching_session] {
        TestRequest({5, 6, 7, 8, 9, 10, 11, 12, 13}, {1, 3, 3},
                    {4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5}, {1, 3, 3},
                    batching_session.get());
      }));
}

TEST_P(BatchingSessionTest, BatchingWithLargeBatch) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 3;
  schedule_options.batch_timeout_micros = 1e6;
  schedule_options.num_batch_threads = 2;
  schedule_options = annotate_options(schedule_options);
  schedule_options.max_execution_batch_size = 2;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));
  if (enable_large_batch_splitting()) {
    // `max_execution_batch_size` is 2, so input of second request will be
    // split for processing.
    std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
        ThreadOptions(), "first_request", [&batching_session] {
          TestRequest({5, 6, 7, 8}, {1, 2, 2}, {4.5, 5, 5.5, 6}, {1, 2, 2},
                      batching_session.get());
        }));
    std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
        ThreadOptions(), "second_request", [&batching_session] {
          TestRequest({5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, {3, 2, 2},
                      {4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10},
                      {3, 2, 2}, batching_session.get());
        }));
  } else {
    Tensor input1 = test::AsTensor<float>({5, 6, 7, 8}, {1, 2, 2});
    Tensor expected_output1 =
        test::AsTensor<float>({4.5, 5, 5.5, 6}, {1, 2, 2});
    std::vector<Tensor> output1;
    Notification notify;
    std::unique_ptr<Thread> first_request_thread(
        Env::Default()->StartThread(ThreadOptions(), "first_request", [&] {
          auto status =
              batching_session->Run({{"x", input1}}, {"y"}, {}, &output1);
          EXPECT_TRUE(status.ok());
          test::ExpectTensorEqual<float>(expected_output1, output1[0]);
        }));

    // `max_batch_size` is 3, so input2 (of size 4) will be invalidated.
    Tensor input2 = test::AsTensor<float>(
        {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, {4, 2, 2});
    std::vector<Tensor> output2;
    std::unique_ptr<Thread> second_request_thread(
        Env::Default()->StartThread(ThreadOptions(), "second_request", [&] {
          auto status =
              batching_session->Run({{"x", input2}}, {"y"}, {}, &output2);
          EXPECT_FALSE(status.ok());
          EXPECT_THAT(status.error_message(),
                      HasSubstr("Task size 4 is larger than "
                                "maximum input batch size 3"));
        }));
  }
}

TEST_P(BatchingSessionTest, BatchHandlesSplitError) {
  if (!enable_large_batch_splitting()) {
    return;
  }

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 3;
  schedule_options.batch_timeout_micros = INT_MAX;  // set a large time out
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  schedule_options.max_execution_batch_size = 2;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));

  string expected_error_msg =
      "Tensors with name 'x' from different tasks have different shapes and "
      "padding is turned off. Set pad_variable_length_inputs to true, or "
      "ensure that all tensors with the same name have equal dimensions "
      "starting with the first dim.";

  // `max_batch_size` is 3 and `max_execution_batch_size` is 2, so inputs of
  // first thread will span over two tasks, causing errors in both batch tasks.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request",
      [&batching_session, &expected_error_msg] {
        ExpectError(expected_error_msg,
                    {{"x", test::AsTensor<float>({1, 2, 3}, {3, 1, 1})}}, {"y"},
                    batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request",
      [&batching_session, &expected_error_msg] {
        ExpectError(expected_error_msg,
                    {{"x", test::AsTensor<float>({1, 2}, {1, 2})}}, {"y"},
                    batching_session.get());
      }));
}

TEST_P(BatchingSessionTest, BatchingLazySplit) {
  if (!enable_large_batch_splitting()) {
    return;
  }

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;
  schedule_options.batch_timeout_micros = INT_MAX;  // set a large time out
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  schedule_options.max_execution_batch_size = 1;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));

  // `max_batch_size` is 2 and `max_execution_batch_size` is 1, so inputs
  // will be split and process.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request", [&batching_session] {
        TestRequest({5, 6, 7, 8}, {1, 2, 2}, {4.5, 5, 5.5, 6}, {1, 2, 2},
                    batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request", [&batching_session] {
        TestRequest({1, 2, 3, 4}, {1, 2, 2}, {2.5, 3, 3.5, 4.0}, {1, 2, 2},
                    batching_session.get());
      }));
}

TEST(BatchingSessionTest, BatchingWithPaddingAndCost) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;
  schedule_options.batch_timeout_micros = 1e6;
  schedule_options.num_batch_threads = 1;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  batching_session_options.pad_variable_length_inputs = true;
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      std::move(batch_size_capturing_session), &batching_session));

  CostGraphDef* cg = batch_size_capturing_session_raw->mutable_cost_graph();
  CostGraphDef_AggregatedCost* ag = cg->add_cost();
  ag->set_cost(7.0);
  ag = cg->add_cost();
  ag->set_dimension("named-cost");
  ag->set_cost(1.0);

  // two requests form a batch and first input gets padded with zeros to match
  // [1, 3, 3] shape that is accepted by the model.
  // if padding doesn't work, test will fail.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request", [&batching_session] {
        Tensor input = test::AsTensor<float>({1, 2, 3, 4}, {1, 2, 2});
        Tensor expected_output = test::AsTensor<float>(
            {2.5, 3, 2.5, 3.5, 4, 2.5, 2.5, 2.5, 2.5}, {1, 3, 3});
        std::vector<Tensor> output;
        RunMetadata run_metadata;
        TF_ASSERT_OK(batching_session->Run({}, {{"x", input}}, {"y"}, {},
                                           &output, &run_metadata));
        ASSERT_EQ(1, output.size());
        test::ExpectTensorEqual<float>(expected_output, output[0]);
        const CostGraphDef& cgs = run_metadata.cost_graph();
        EXPECT_EQ(2, cgs.cost_size());
        EXPECT_NEAR(3.5, cgs.cost(0).cost(), 0.001);
        EXPECT_NEAR(0.5, cgs.cost(1).cost(), 0.001);
        EXPECT_EQ("named-cost", cgs.cost(1).dimension());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request", [&batching_session] {
        Tensor input =
            test::AsTensor<float>({5, 6, 7, 8, 9, 10, 11, 12, 13}, {1, 3, 3});
        Tensor expected_output = test::AsTensor<float>(
            {4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5}, {1, 3, 3});
        std::vector<Tensor> output;
        RunMetadata run_metadata;
        TF_ASSERT_OK(batching_session->Run({}, {{"x", input}}, {"y"}, {},
                                           &output, &run_metadata));
        ASSERT_EQ(1, output.size());
        test::ExpectTensorEqual<float>(expected_output, output[0]);
        const CostGraphDef& cgs = run_metadata.cost_graph();
        EXPECT_EQ(2, cgs.cost_size());
        EXPECT_NEAR(3.5, cgs.cost(0).cost(), 0.001);
        EXPECT_NEAR(0.5, cgs.cost(1).cost(), 0.001);
        EXPECT_EQ("named-cost", cgs.cost(1).dimension());
      }));
}

TEST_P(BatchingSessionTest, BatchingWithCost) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 3;
  schedule_options.batch_timeout_micros = 1e6;
  schedule_options.num_batch_threads = 2;
  schedule_options = annotate_options(schedule_options);
  schedule_options.max_execution_batch_size = 2;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      std::move(batch_size_capturing_session), &batching_session));

  CostGraphDef* cg = batch_size_capturing_session_raw->mutable_cost_graph();
  CostGraphDef_AggregatedCost* ag = cg->add_cost();
  ag->set_cost(7.0);
  ag = cg->add_cost();
  ag->set_dimension("named-cost");
  ag->set_cost(1.0);

  // two requests form a batch and first input gets padded with zeros to match
  // [1, 3, 3] shape that is accepted by the model.
  // if padding doesn't work, test will fail.
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request", [&batching_session] {
        Tensor input = test::AsTensor<float>({1, 2, 3, 4, 5, 6}, {1, 2, 3});
        Tensor expected_output =
            test::AsTensor<float>({2.5, 3, 3.5, 4, 4.5, 5}, {1, 2, 3});
        std::vector<Tensor> output;
        RunMetadata run_metadata;
        TF_ASSERT_OK(batching_session->Run({}, {{"x", input}}, {"y"}, {},
                                           &output, &run_metadata));
        ASSERT_EQ(1, output.size());
        test::ExpectTensorEqual<float>(expected_output, output[0]);
        const CostGraphDef& cgs = run_metadata.cost_graph();
        EXPECT_EQ(2, cgs.cost_size());
        EXPECT_NEAR(3.5, cgs.cost(0).cost(), 0.001);
        EXPECT_NEAR(0.5, cgs.cost(1).cost(), 0.001);
        EXPECT_EQ("named-cost", cgs.cost(1).dimension());
      }));
  if (enable_large_batch_splitting()) {
    std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
        ThreadOptions(), "second_request", [&batching_session] {
          Tensor input =
              test::AsTensor<float>({5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                     17, 18, 19, 20, 21, 22},
                                    {3, 2, 3});
          Tensor expected_output =
              test::AsTensor<float>({4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9,
                                     9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13},
                                    {3, 2, 3});
          std::vector<Tensor> output;
          RunMetadata run_metadata;
          TF_ASSERT_OK(batching_session->Run({}, {{"x", input}}, {"y"}, {},
                                             &output, &run_metadata));
          ASSERT_EQ(1, output.size());
          test::ExpectTensorEqual<float>(expected_output, output[0]);
          const CostGraphDef& cgs = run_metadata.cost_graph();
          EXPECT_EQ(2, cgs.cost_size());
          EXPECT_NEAR(10.5, cgs.cost(0).cost(), 0.001);
          EXPECT_NEAR(1.5, cgs.cost(1).cost(), 0.001);
          EXPECT_EQ("named-cost", cgs.cost(1).dimension());
        }));
  } else {
    std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
        ThreadOptions(), "second_request", [&batching_session] {
          Tensor input = test::AsTensor<float>({5, 6, 7, 8, 9, 10}, {1, 2, 3});
          Tensor expected_output =
              test::AsTensor<float>({4.5, 5, 5.5, 6, 6.5, 7}, {1, 2, 3});
          std::vector<Tensor> output;
          RunMetadata run_metadata;
          TF_ASSERT_OK(batching_session->Run({}, {{"x", input}}, {"y"}, {},
                                             &output, &run_metadata));
          ASSERT_EQ(1, output.size());
          test::ExpectTensorEqual<float>(expected_output, output[0]);
          const CostGraphDef& cgs = run_metadata.cost_graph();
          EXPECT_EQ(2, cgs.cost_size());
          EXPECT_NEAR(3.5, cgs.cost(0).cost(), 0.001);
          EXPECT_NEAR(0.5, cgs.cost(1).cost(), 0.001);
          EXPECT_EQ("named-cost", cgs.cost(1).dimension());
        }));
  }
}

TEST_P(BatchingSessionTest, UnequalTensorShapesWithPaddingTurnedOff) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 2;
  schedule_options.batch_timeout_micros = 1e6;
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  batching_session_options.pad_variable_length_inputs = false;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateMatrixHalfPlusTwoSession(), &batching_session));
  string expected_error_msg =
      "Tensors with name 'x' from different tasks have different shapes and "
      "padding is turned off. Set pad_variable_length_inputs to true, or "
      "ensure that all tensors with the same name have equal dimensions "
      "starting with the first dim.";
  std::unique_ptr<Thread> first_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request",
      [&batching_session, &expected_error_msg] {
        ExpectError(expected_error_msg,
                    {{"x", test::AsTensor<float>({1, 2, 3, 4}, {1, 2, 2})}},
                    {"y"}, batching_session.get());
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "first_request",
      [&batching_session, &expected_error_msg] {
        ExpectError(expected_error_msg,
                    {{"x", test::AsTensor<float>(
                               {5, 6, 7, 8, 9, 10, 11, 12, 13}, {1, 3, 3})}},
                    {"y"}, batching_session.get());
      }));
}

TEST_P(BatchingSessionTest, SingletonBatch) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;  // fits two 2-unit tasks
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));
  TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
              batching_session.get());
}

TEST_P(BatchingSessionTest, RequestThatDoesntMatchSignatureGetsRunAnyway) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  // Set the batching parameters s.t. if the request is batched the test will
  // timeout.
  schedule_options.max_batch_size = 100;
  schedule_options.batch_timeout_micros = INT_MAX;
  schedule_options = annotate_options(schedule_options);
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x2"}, {"y3"}},
      CreateHalfPlusTwoSession(), &batching_session));
  // Issue a request using x/y, which doesn't match the x2/y3 signature.
  TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
              batching_session.get());
}

TEST_P(BatchingSessionTest, RequestWithIncompatibleInputTensorSizes) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options = annotate_options(schedule_options);
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;

  int32 start_input_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/input_batch_size");
  int32 start_process_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/processed_batch_size");
  int32 start_pad_value =
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size");
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options,
      {{"input_0", "input_1"}, {"output"}}, CreateHalfPlusTwoSession(),
      &batching_session));

  ExpectError("Batching Run() input tensors must have equal 0th-dimension size",
              {{"input_0", test::AsTensor<int>({3}, {1})},
               {"input_1", test::AsTensor<int>({5, 7}, {2})}},
              {"output"}, batching_session.get());

  // We expect no change.
  EXPECT_EQ(start_input_value,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/input_batch_size"));
  EXPECT_EQ(start_process_value,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/processed_batch_size"));
  EXPECT_EQ(
      start_pad_value,
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size"));
}

TEST_P(BatchingSessionTest, AllowedBatchSizesNoPaddingNeeded) {
  int32 start_input_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/input_batch_size");
  int32 start_process_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/processed_batch_size");
  int32 start_pad_value =
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size");
  // Arrange to capture the batch size.
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      std::move(batch_size_capturing_session), &batching_session));
  TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
              batching_session.get());

  // It should not add any padding, i.e. leave the batch size at 2.
  EXPECT_EQ(2, batch_size_capturing_session_raw->latest_batch_size());

  // We expect no pad, 2 inputs, and a batch process of 2.
  EXPECT_EQ(start_input_value + 2,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/input_batch_size"));
  EXPECT_EQ(start_process_value + 2,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/processed_batch_size"));
  EXPECT_EQ(
      start_pad_value,
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size"));
}

TEST_P(BatchingSessionTest, AllowedBatchSizesRequirePadding) {
  int32 start_input_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/input_batch_size");
  int32 start_process_value = GetPercentileTotal(
      "/tensorflow/serving/batching_session/processed_batch_size");
  int32 start_pad_value =
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size");

  // Arrange to capture the batch size.
  std::unique_ptr<BatchSizeCapturingSession> batch_size_capturing_session(
      new BatchSizeCapturingSession(CreateHalfPlusTwoSession()));
  auto batch_size_capturing_session_raw = batch_size_capturing_session.get();

  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options.batch_timeout_micros = 0;
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {1, 3, 4};
  std::unique_ptr<Session> batching_session;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      std::move(batch_size_capturing_session), &batching_session));
  TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
              batching_session.get());

  // It should pad the batch size from 2 to 3.
  EXPECT_EQ(3, batch_size_capturing_session_raw->latest_batch_size());

  // We expect 1 pad, 2 inputs, and a batch process of 3.
  EXPECT_EQ(start_input_value + 2,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/input_batch_size"));
  EXPECT_EQ(start_process_value + 3,
            GetPercentileTotal(
                "/tensorflow/serving/batching_session/processed_batch_size"));
  EXPECT_EQ(
      start_pad_value + 1,
      GetPercentileTotal("/tensorflow/serving/batching_session/padding_size"));
  EXPECT_TRUE(
      CheckDescriptor("/tensorflow/serving/batching_session/padding_size",
                      "Tracks the padding size distribution on batches.",
                      {"execution_batch_size"}));
  EXPECT_TRUE(
      CheckDescriptor("/tensorflow/serving/batching_session/input_batch_size",
                      "Tracks the batch size distribution on the inputs.", {}));
  EXPECT_TRUE(CheckDescriptor(
      "/tensorflow/serving/batching_session/processed_batch_size",
      "Tracks the batch size distribution on processing.", {}));
}

TEST_P(BatchingSessionTest, UnsortedAllowedBatchSizesRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options = annotate_options(schedule_options);
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {4, 2};  // Not sorted.
  std::unique_ptr<Session> batching_session;
  EXPECT_FALSE(CreateBasicBatchingSession(
                   schedule_options, batching_session_options, {{"x"}, {"y"}},
                   CreateHalfPlusTwoSession(), &batching_session)
                   .ok());
}

TEST_P(BatchingSessionTest,
       FinalAllowedBatchSizeLargerThanMaxBatchSizeRejected) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 4;
  schedule_options = annotate_options(schedule_options);
  BatchingSessionOptions batching_session_options;
  batching_session_options.allowed_batch_sizes = {2, 8};  // Final entry != 4.
  std::unique_ptr<Session> batching_session;
  auto status = CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), HasSubstr(enable_large_batch_splitting()
                                                    ? "max_execution_batch_size"
                                                    : "max_batch_size"));
}

TEST_P(BatchingSessionTest, DifferentOrderForInputAndOutputTensors) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 6;  // fits three 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
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

TEST_P(BatchingSessionTest, MultipleSignatures) {
  std::vector<BatchScheduler<BatchingSessionTask>*> schedulers;
  auto create_scheduler =
      [&schedulers, this](
          std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
              process_batch_callback,
          std::unique_ptr<BatchScheduler<BatchingSessionTask>>* scheduler) {
        BasicBatchScheduler<BatchingSessionTask>::Options options;
        options.max_batch_size = 4;  // fits two 2-unit tasks
        options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
        options.num_batch_threads = 1;
        options = annotate_options(options);
        std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>>
            basic_scheduler;
        TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
            options, process_batch_callback, &basic_scheduler));
        schedulers.push_back(basic_scheduler.get());
        *scheduler = std::move(basic_scheduler);
        return OkStatus();
      };
  BatchingSessionOptions batching_session_options;
  std::unique_ptr<Session> batching_session;
  TF_CHECK_OK(CreateBatchingSession(batching_session_options,
                                    {{{{"x"}, {"y"}}, create_scheduler},
                                     {{{"x2"}, {"y3"}}, create_scheduler}},
                                    CreateHalfPlusTwoSession(),
                                    &batching_session));
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

TEST_P(BatchingSessionTest, EnqueuedLongerThanTimeout) {
  BatchScheduler<BatchingSessionTask>* scheduler = nullptr;
  auto create_scheduler =
      [&scheduler, this](
          std::function<void(std::unique_ptr<Batch<BatchingSessionTask>>)>
              process_batch_callback,
          std::unique_ptr<BatchScheduler<BatchingSessionTask>>* new_scheduler) {
        BasicBatchScheduler<BatchingSessionTask>::Options options;
        options.max_batch_size = 4;  // fits two 2-unit tasks
        options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
        options.num_batch_threads = 1;
        options = annotate_options(options);
        std::unique_ptr<BasicBatchScheduler<BatchingSessionTask>>
            basic_scheduler;
        TF_RETURN_IF_ERROR(BasicBatchScheduler<BatchingSessionTask>::Create(
            options, process_batch_callback, &basic_scheduler));
        scheduler = basic_scheduler.get();
        *new_scheduler = std::move(basic_scheduler);
        return OkStatus();
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

TEST_P(BatchingSessionTest, ThreadPoolOptions) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 3;
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
  schedule_options.max_execution_batch_size = 1;
  std::unique_ptr<Session> batching_session;
  BatchingSessionOptions batching_session_options;
  TF_ASSERT_OK(CreateBasicBatchingSession(
      schedule_options, batching_session_options, {{"x"}, {"y"}},
      CreateHalfPlusTwoSession(), &batching_session));

  test_util::CountingThreadPool inter_op_threadpool(Env::Default(), "InterOp",
                                                    /*num_threads=*/1);
  test_util::CountingThreadPool intra_op_threadpool(Env::Default(), "IntraOp",
                                                    /*num_threads=*/1);

  // Asynchronously send two requests whose total size is 4.
  // They form two batches in both non-split and input-split mode.
  std::unique_ptr<Thread> first_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "first_request_thread", [&] {
        TestRequest({100.0f, 42.0f}, {2}, {52.0f, 23.0f}, {2},
                    batching_session.get(), &inter_op_threadpool,
                    &intra_op_threadpool);
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        TestRequest({71.5f, 18.3f}, {2}, {37.75f, 11.15f}, {2},
                    batching_session.get(), &inter_op_threadpool,
                    &intra_op_threadpool);
      }));
}

TEST_P(BatchingSessionTest, SubsetOutputTensors) {
  BasicBatchScheduler<BatchingSessionTask>::Options schedule_options;
  schedule_options.max_batch_size = 6;  // fits three 2-unit tasks
  schedule_options.batch_timeout_micros = 1 * 1000 * 1000;  // won't trigger
  schedule_options.num_batch_threads = 1;
  schedule_options = annotate_options(schedule_options);
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
                                           {"y"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(1, outputs.size());
        test::ExpectTensorEqual<float>(expected_output0, outputs[0]);
      }));
  std::unique_ptr<Thread> second_request_thread(Env::Default()->StartThread(
      ThreadOptions(), "second_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(batching_session->Run({{"x2", input1}, {"x", input0}},
                                           {"y3"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(1, outputs.size());
        test::ExpectTensorEqual<float>(expected_output1, outputs[0]);
      }));
  std::unique_ptr<Thread> third_request_thread(
      Env::Default()->StartThread(ThreadOptions(), "third_request_thread", [&] {
        std::vector<Tensor> outputs;
        TF_ASSERT_OK(batching_session->Run({{"x2", input1}, {"x", input0}},
                                           {"y"} /* outputs */,
                                           {} /* target nodes */, &outputs));
        ASSERT_EQ(1, outputs.size());
        test::ExpectTensorEqual<float>(expected_output0, outputs[0]);
      }));
}

INSTANTIATE_TEST_SUITE_P(
    Parameter, BatchingSessionTest,
    ::testing::Values(std::make_tuple(/*enable_input_batch_split=*/false,
                                      /*enable_lazy_split=*/false),
                      std::make_tuple(/*enable_input_batch_split=*/true,
                                      /*enable_lazy_split=*/false),
                      std::make_tuple(/*enable_input_batch_split=*/true,
                                      /*enable_lazy_split=*/true)));

}  // namespace
}  // namespace serving
}  // namespace tensorflow
