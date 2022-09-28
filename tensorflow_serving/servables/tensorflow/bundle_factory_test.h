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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

using test_util::EqualsProto;

// The base class for SessionBundleFactoryTest and SavedModelBundleFactoryTest.
class BundleFactoryTest : public ::testing::Test {
 public:
  explicit BundleFactoryTest(const string &export_dir)
      : export_dir_(export_dir) {}

  virtual ~BundleFactoryTest() = default;

 protected:
  // Test functions to be used by subclasses.
  void TestBasic() const {
    const SessionBundleConfig config = GetSessionBundleConfig();
    std::unique_ptr<Session> session;
    if (ExpectCreateBundleFailure()) {
      EXPECT_FALSE(CreateSession(config, &session).ok());
      return;
    }
    TF_ASSERT_OK(CreateSession(config, &session));
    TestSingleRequest(session.get());
  }

  int GetTotalBatchesProcessed() const {
    const string label(
        "/tensorflow/serving/batching_session/wrapped_run_count");
    auto* collection_registry = monitoring::CollectionRegistry::Default();
    monitoring::CollectionRegistry::CollectMetricsOptions options;
    const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);
    int total_count = 0;
    const auto& point_set_map = collected_metrics->point_set_map;
    if (point_set_map.find(label) == point_set_map.end()) return 0;
    const monitoring::PointSet& lps = *point_set_map.at(label);
    for (int i = 0; i < lps.points.size(); ++i) {
      total_count += lps.points[i]->int64_value;
    }
    return static_cast<int>(total_count);
  }

  void TestBatching(const BatchingParameters& params,
                    bool enable_per_model_batching_params,
                    int input_request_batch_size, int batch_size) const {
    SessionBundleConfig config = GetSessionBundleConfig();
    config.set_enable_per_model_batching_params(
        enable_per_model_batching_params);
    BatchingParameters* batching_params = config.mutable_batching_parameters();
    *batching_params = params;

    //
    // Tweak batching params further for testing.
    //
    // Set high value of max enqueued batches to prevent queue limits to be hit
    // during testing that involves lot of requests.
    batching_params->mutable_max_enqueued_batches()->set_value(INT_MAX);
    // Set very high value of timeout to force full batches to be formed.
    //
    // The default (zero) value of the timeout causes batches to formed with
    // [1..max_batch_size] size based on relative ordering of Run() calls. A
    // large value causes deterministic fix batch size to be formed.
    batching_params->mutable_batch_timeout_micros()->set_value(INT_MAX);

    std::unique_ptr<Session> session;
    if (ExpectCreateBundleFailure()) {
      EXPECT_FALSE(CreateSession(config, &session).ok());
      return;
    }
    TF_ASSERT_OK(CreateSession(config, &session));

    const int num_requests = 10;
    const int expected_batches =
        (input_request_batch_size * num_requests) / batch_size;
    const int orig_batches_processed = GetTotalBatchesProcessed();
    TestMultipleRequests(session.get(), num_requests, input_request_batch_size);
    EXPECT_EQ(orig_batches_processed + expected_batches,
              GetTotalBatchesProcessed());
  }

  template <class FactoryType>
  void TestEstimateResourceRequirementWithGoodExport(
      double total_file_size) const {
    const SessionBundleConfig config = GetSessionBundleConfig();
    std::unique_ptr<FactoryType> factory;
    TF_ASSERT_OK(FactoryType::Create(config, &factory));
    ResourceAllocation actual;
    TF_ASSERT_OK(factory->EstimateResourceRequirement(export_dir_, &actual));

    ResourceAllocation expected = GetExpectedResourceEstimate(total_file_size);
    EXPECT_THAT(actual, EqualsProto(expected));
  }

  void TestRunOptions() const {
    if (!IsRunOptionsSupported()) return;

    SessionBundleConfig config = GetSessionBundleConfig();

    // Configure the session-config with two threadpools. The first is setup
    // with default settings. The second is explicitly setup with 1 thread.
    config.mutable_session_config()->add_session_inter_op_thread_pool();
    config.mutable_session_config()
        ->add_session_inter_op_thread_pool()
        ->set_num_threads(1);

    // Set the threadpool index to use for session-run calls to 1.
    config.mutable_session_run_load_threadpool_index()->set_value(1);

    // Since the session_run_load_threadpool_index in the config is set, the
    // session-bundle should be loaded successfully from path with RunOptions.
    std::unique_ptr<Session> session;
    if (ExpectCreateBundleFailure()) {
      EXPECT_FALSE(CreateSession(config, &session).ok());
      return;
    }
    TF_ASSERT_OK(CreateSession(config, &session));

    TestSingleRequest(session.get());
  }

  void TestRunOptionsError() const {
    if (!IsRunOptionsSupported()) return;

    // Session bundle config with the default global threadpool.
    SessionBundleConfig config = GetSessionBundleConfig();

    // Invalid threadpool index to use for session-run calls.
    config.mutable_session_run_load_threadpool_index()->set_value(100);

    // Since RunOptions used in the session run calls refers to an invalid
    // threadpool index, load session bundle from path should fail.
    std::unique_ptr<Session> session;
    EXPECT_FALSE(CreateSession(config, &session).ok());
  }

  // Test data path, to be initialized to point at a SessionBundle export or
  // SavedModel of half-plus-two.
  string export_dir_;

 private:
  // Creates a Session with the given configuration and export path.
  virtual Status CreateSession(const SessionBundleConfig &config,
                               std::unique_ptr<Session> *session) const = 0;

  // Returns a SessionBundleConfig.
  virtual SessionBundleConfig GetSessionBundleConfig() const {
    return SessionBundleConfig();
  }

  // Returns true if RunOptions is supported by underlying session.
  virtual bool IsRunOptionsSupported() const { return true; }

  // Returns true if CreateBundle is expected to fail.
  virtual bool ExpectCreateBundleFailure() const { return false; }
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_H_
