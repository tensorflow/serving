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

#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle_util.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/test_util/mock_file_probing_env.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

using Batcher = SharedBatchScheduler<BatchingSessionTask>;

class BundleFactoryUtilTest : public ::testing::Test {
 protected:
  BundleFactoryUtilTest() : export_dir_(test_util::GetTestSavedModelPath()) {}

  virtual ~BundleFactoryUtilTest() = default;

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;
};

TEST_F(BundleFactoryUtilTest, GetSessionOptions) {
  SessionBundleConfig bundle_config;

  constexpr char kTarget[] = "target";
  bundle_config.set_session_target(kTarget);
  ConfigProto *config_proto = bundle_config.mutable_session_config();
  config_proto->set_allow_soft_placement(true);

  SessionOptions session_options = GetSessionOptions(bundle_config);
  EXPECT_EQ(session_options.target, kTarget);
  EXPECT_THAT(session_options.config, EqualsProto(*config_proto));
}

TEST_F(BundleFactoryUtilTest, GetRunOptions) {
  SessionBundleConfig bundle_config;

  // Set the threadpool index to use for session-run calls to 1.
  bundle_config.mutable_session_run_load_threadpool_index()->set_value(1);

  RunOptions want;
  want.set_inter_op_thread_pool(1);
  EXPECT_THAT(GetRunOptions(bundle_config), EqualsProto(want));
}

TEST_F(BundleFactoryUtilTest, WrapSession) {
  SavedModelBundle bundle;
  TF_ASSERT_OK(LoadSavedModel(SessionOptions(), RunOptions(), export_dir_,
                              {"serve"}, &bundle));
  TF_ASSERT_OK(WrapSession(&bundle.session));
  test_util::TestSingleRequest(bundle.session.get());
}

TEST_F(BundleFactoryUtilTest, WrapSessionForBatching) {
  SavedModelBundle bundle;
  TF_ASSERT_OK(LoadSavedModel(SessionOptions(), RunOptions(), export_dir_,
                              {"serve"}, &bundle));

  // Create BatchingParameters and batch scheduler.
  BatchingParameters batching_params;
  batching_params.mutable_max_batch_size()->set_value(2);
  batching_params.mutable_max_enqueued_batches()->set_value(INT_MAX);

  std::shared_ptr<Batcher> batcher;
  TF_ASSERT_OK(CreateBatchScheduler(batching_params, &batcher));

  // Wrap the session.
  TF_ASSERT_OK(WrapSessionForBatching(batching_params, batcher,
                                      {test_util::GetTestSessionSignature()},
                                      &bundle.session));

  // Run multiple requests concurrently. They should be executed as 5 batches.
  test_util::TestMultipleRequests(10, bundle.session.get());
}

TEST_F(BundleFactoryUtilTest, BatchingConfigError) {
  BatchingParameters batching_params;
  batching_params.mutable_max_batch_size()->set_value(2);
  // The last entry in 'allowed_batch_sizes' is supposed to equal
  // 'max_batch_size'. Let's violate that constraint and ensure we get an error.
  batching_params.add_allowed_batch_sizes(1);
  batching_params.add_allowed_batch_sizes(3);
  std::shared_ptr<Batcher> batch_scheduler;
  EXPECT_FALSE(CreateBatchScheduler(batching_params, &batch_scheduler).ok());
}

TEST_F(BundleFactoryUtilTest, EstimateResourceFromPathWithBadExport) {
  ResourceAllocation resource_requirement;
  const Status status = EstimateResourceFromPath(
      "/a/bogus/export/dir",
      /*use_validation_result=*/false, &resource_requirement);
  EXPECT_FALSE(status.ok());
}

TEST_F(BundleFactoryUtilTest, EstimateResourceFromPathWithGoodExport) {
  const double kTotalFileSize = test_util::GetTotalFileSize(
      test_util::GetTestSavedModelBundleExportFiles());
  ResourceAllocation expected =
      test_util::GetExpectedResourceEstimate(kTotalFileSize);

  ResourceAllocation actual;
  TF_ASSERT_OK(EstimateResourceFromPath(
      export_dir_, /*use_validation_result=*/false, &actual));
  EXPECT_THAT(actual, EqualsProto(expected));
}

#ifdef PLATFORM_GOOGLE
// This benchmark relies on https://github.com/google/benchmark features,
// not available in open-sourced TF codebase.

void BM_HalfPlusTwo(benchmark::State& state) {
  static Session* session;
  if (state.thread_index() == 0) {
    SavedModelBundle bundle;
    TF_ASSERT_OK(LoadSavedModel(SessionOptions(), RunOptions(),
                                test_util::GetTestSavedModelPath(), {"serve"},
                                &bundle));
    TF_ASSERT_OK(WrapSession(&bundle.session));
    session = bundle.session.release();
  }
  Tensor input = test::AsTensor<float>({1.0, 2.0, 3.0}, TensorShape({3}));
  std::vector<Tensor> outputs;
  for (auto _ : state) {
    outputs.clear();
    TF_ASSERT_OK(session->Run({{"x:0", input}}, {"y:0"}, {}, &outputs));
  }
}
BENCHMARK(BM_HalfPlusTwo)->UseRealTime()->ThreadRange(1, 64);

#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace serving
}  // namespace tensorflow
