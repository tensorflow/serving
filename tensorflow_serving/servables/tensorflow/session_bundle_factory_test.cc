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

#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class SessionBundleFactoryTest : public test_util::BundleFactoryTest {
 public:
  virtual ~SessionBundleFactoryTest() = default;
};

TEST_F(SessionBundleFactoryTest, Basic) {
  const SessionBundleConfig config;
  std::unique_ptr<SessionBundleFactory> factory;
  TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));
  std::unique_ptr<SessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));
  TestSingleRequest(bundle->session.get());
}

TEST_F(SessionBundleFactoryTest, Batching) {
  SessionBundleConfig config;
  BatchingParameters* batching_params = config.mutable_batching_parameters();
  batching_params->mutable_max_batch_size()->set_value(2);
  batching_params->mutable_max_enqueued_batches()->set_value(INT_MAX);
  std::unique_ptr<SessionBundleFactory> factory;
  TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));
  std::unique_ptr<SessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));

  // Run multiple requests concurrently. They should be executed as 5 batches,
  // as request size is set to 2 in TestMultipleRequests().
  TestMultipleRequests(10, bundle->session.get());
}

TEST_F(SessionBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
  // The length of the file's version strings might change, so we don't
  // hardcode their size.  They are 4 bytes for tags & size, plus the actual
  // length of the strings.
  const double kVersionSize =
      4 + strlen(TF_VERSION_STRING) + strlen(tf_git_version());
  const double kTotalFileSize = 13392.5 + kVersionSize;
  ResourceAllocation expected = GetExpectedResourceEstimate(kTotalFileSize);

  const SessionBundleConfig config;
  std::unique_ptr<SessionBundleFactory> factory;
  TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));
  ResourceAllocation actual;
  TF_ASSERT_OK(factory->EstimateResourceRequirement(export_dir_, &actual));

  EXPECT_THAT(actual, EqualsProto(expected));
}

TEST_F(SessionBundleFactoryTest, RunOptions) {
  SessionBundleConfig config;

  // Configure the session-config with two threadpools. The first is setup with
  // default settings. The second is explicitly setup with 1 thread.
  config.mutable_session_config()->add_session_inter_op_thread_pool();
  config.mutable_session_config()
      ->add_session_inter_op_thread_pool()
      ->set_num_threads(1);

  // Set the threadpool index to use for session-run calls to 1.
  config.mutable_session_run_load_threadpool_index()->set_value(1);

  std::unique_ptr<SessionBundleFactory> factory;
  TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));

  // Since the session_run_load_threadpool_index in the config is set, the
  // session-bundle should be loaded successfully from path with RunOptions.
  std::unique_ptr<SessionBundle> bundle;
  TF_ASSERT_OK(factory->CreateSessionBundle(export_dir_, &bundle));

  TestSingleRequest(bundle->session.get());
}

TEST_F(SessionBundleFactoryTest, RunOptionsError) {
  // Session bundle config with the default global threadpool.
  SessionBundleConfig config;

  // Invalid threadpool index to use for session-run calls.
  config.mutable_session_run_load_threadpool_index()->set_value(100);

  std::unique_ptr<SessionBundleFactory> factory;
  TF_ASSERT_OK(SessionBundleFactory::Create(config, &factory));

  // Since RunOptions used in the session run calls refers to an invalid
  // threadpool index, load session bundle from path should fail.
  std::unique_ptr<SessionBundle> bundle;
  EXPECT_FALSE(factory->CreateSessionBundle(export_dir_, &bundle).ok());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
