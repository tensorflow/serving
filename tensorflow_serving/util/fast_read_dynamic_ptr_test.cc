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

#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {
namespace {

template <typename T>
class FastReadDynamicPtrTest : public ::testing::Test {};

using FastReadDynamicPtrTypes = ::testing::Types<
    FastReadDynamicPtr<int>,
    FastReadDynamicPtr<int, internal_read_ptr_holder::ShardedReadPtrs<int>>,
    FastReadDynamicPtr<int, internal_read_ptr_holder::SingleReadPtr<int>>>;

TYPED_TEST_SUITE(FastReadDynamicPtrTest, FastReadDynamicPtrTypes);

TYPED_TEST(FastReadDynamicPtrTest, SingleThreaded) {
  TypeParam fast_read_int;

  {
    // Initially the object should be null.
    std::shared_ptr<const int> pointer = fast_read_int.get();
    EXPECT_EQ(pointer, nullptr);
  }

  // Swap in an actual value.
  std::unique_ptr<int> i(new int(1));
  fast_read_int.Update(std::move(i));
  EXPECT_EQ(nullptr, i);

  {
    std::shared_ptr<const int> pointer = fast_read_int.get();
    EXPECT_EQ(*pointer, 1);
  }
}

TYPED_TEST(FastReadDynamicPtrTest, MultiThreaded) {
  const int kNumThreads = 4;

  TypeParam fast_read_int;

  {
    std::unique_ptr<int> tmp(new int(0));
    EXPECT_EQ(nullptr, fast_read_int.Update(std::move(tmp)));
  }

  std::vector<std::unique_ptr<Thread>> threads;
  for (int thread_index = 0; thread_index < kNumThreads; ++thread_index) {
    // Spawn a new thread.
    threads.emplace_back(Env::Default()->StartThread(
        {}, "Increment", [thread_index, &fast_read_int]() {
          const int kMaxValue = 1000;
          int last_value = -1;
          // Loops until 'fast_read_int' becomes 'max_value'.
          for (;;) {
            int value = -1;
            {
              std::shared_ptr<const int> pointer = fast_read_int.get();
              value = *pointer;
            }

            EXPECT_GE(value, last_value);
            if (value == kMaxValue) {
              return;
            }
            if (value % kNumThreads == thread_index) {
              std::unique_ptr<int> tmp(new int(value + 1));
              fast_read_int.Update(std::move(tmp));
            }
          }
        }));
  }
}

TYPED_TEST(FastReadDynamicPtrTest, WaitsForReadPtrsBeforeDestruction) {
  const int expected = 12;
  std::unique_ptr<TypeParam> fast_read_int(new TypeParam);
  fast_read_int->Update(std::unique_ptr<int>(new int(expected)));
  Notification got;
  std::unique_ptr<Thread> thread(Env::Default()->StartThread({}, "Holder", [&] {
    auto p = fast_read_int->get();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(expected, *p);
    got.Notify();
    // The other thread can't notify us that they deleted
    // fast_read_int because that destruction will block until
    // we're done with p.  We sleep so that with high
    // probability, deletion was attempted.
    Env::Default()->SleepForMicroseconds(1e7);
    EXPECT_EQ(expected, *p);
  }));
  got.WaitForNotification();
  // Destruction must block until all outstanding ReadPtrs are destroyed.
  fast_read_int = nullptr;
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
