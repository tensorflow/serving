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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Base class of tests related to bundle factories. It contains functions to
// run requests for a half plus model and estimate its resource usage.
class BundleFactoryTest : public ::testing::Test {
 public:
  virtual ~BundleFactoryTest() = default;

 protected:
  BundleFactoryTest()
      : export_dir_(test_util::ContribTestSrcDirPath(
            "session_bundle/example/half_plus_two/00000123")) {}

  // Test that a Session handles a single request for the half plus two
  // model properly. The request has size=2, for batching purposes.
  void TestSingleRequest(Session* session) const;

  // Test that a Session handles multiple concurrent requests for the half plus
  // two model properly. The request has size=2, for batching purposes.
  void TestMultipleRequests(int num_requests, Session* session) const;

  // Returns the expected resource estimate for the given total file size.
  ResourceAllocation GetExpectedResourceEstimate(double total_file_size) const;

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_TEST_UTIL_H_
