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

#include "tensorflow_serving/util/retrier.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;

TEST(RetrierTest, RetryFinallySucceeds) {
  auto retried_fn = []() {
    static int count = 0;
    ++count;
    if (count == 1) {
      return errors::Unknown("Error");
    }
    return Status::OK();
  };

  TF_EXPECT_OK(Retry("RetryFinallySucceeds", 1 /* max_num_retries */,
                     1 /* retry_interval_micros */, retried_fn));
}

TEST(RetrierTest, RetryFinallyFails) {
  auto retried_fn = []() {
    static int count = 0;
    if (++count <= 2) {
      return errors::Unknown("Error");
    }
    return Status::OK();
  };

  const auto status = Retry("RetryFinallyFails", 1 /* max_num_retries */,
                            0 /* retry_interval_micros */, retried_fn);
  EXPECT_THAT(status.error_message(), HasSubstr("Error"));
}

TEST(RetrierTest, RetryCancelled) {
  int call_count = 0;
  auto retried_fn = [&]() {
    ++call_count;
    return errors::Unknown("Error");
  };
  const auto status = Retry("RetryCancelled", 10 /* max_num_retries */,
                            0 /* retry_interval_micros */, retried_fn,
                            []() { return true; } /* cancelled */);
  EXPECT_THAT(status.error_message(), HasSubstr("Error"));
  EXPECT_EQ(1, call_count);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
