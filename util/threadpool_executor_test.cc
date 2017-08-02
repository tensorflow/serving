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

#include "tensorflow_serving/util/threadpool_executor.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

constexpr int kNumThreads = 30;

TEST(ThreadPoolExecutor, Empty) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    LOG(INFO) << "Testing with " << num_threads << " threads";
    ThreadPoolExecutor pool(Env::Default(), "test", num_threads);
  }
}

TEST(ThreadPoolExecutor, DoWork) {
  for (int num_threads = 1; num_threads < kNumThreads; num_threads++) {
    LOG(INFO) << "Testing with " << num_threads << " threads";
    const int kWorkItems = 15;
    // Not using std::vector<bool> due to its unusual implementation and API -
    // http://en.cppreference.com/w/cpp/container/vector_bool
    bool work[kWorkItems];
    for (int i = 0; i < kWorkItems; ++i) {
      work[i] = false;
    }
    {
      ThreadPoolExecutor executor(Env::Default(), "test", num_threads);
      for (int i = 0; i < kWorkItems; i++) {
        executor.Schedule([&work, i]() {
          ASSERT_FALSE(work[i]);
          work[i] = true;
        });
      }
    }
    for (int i = 0; i < kWorkItems; i++) {
      ASSERT_TRUE(work[i]);
    }
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
