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

#include "tensorflow_serving/util/shared_counter.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(SharedCounterTest, OneThread) {
  SharedCounter counter;
  EXPECT_EQ(0, counter.GetValue());
  counter.Increment();
  EXPECT_EQ(1, counter.GetValue());
  counter.Increment();
  EXPECT_EQ(2, counter.GetValue());
  counter.Decrement();
  EXPECT_EQ(1, counter.GetValue());
  counter.Increment();
  EXPECT_EQ(2, counter.GetValue());
}

enum class OperationType { kIncrement, kDecrement };

TEST(SharedCounterTest, Waiting) {
  for (const OperationType operation_type :
       {OperationType::kIncrement, OperationType::kDecrement}) {
    SharedCounter counter;
    counter.Increment();
    Notification done_waiting;
    std::unique_ptr<Thread> thread(
        Env::Default()->StartThread({}, "Waiter", [&counter, &done_waiting]() {
          counter.WaitUntilChanged();
          done_waiting.Notify();
        }));
    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100 milliseconds */);
    EXPECT_FALSE(done_waiting.HasBeenNotified());
    switch (operation_type) {
      case OperationType::kIncrement:
        counter.Increment();
        break;
      case OperationType::kDecrement:
        counter.Decrement();
        break;
    }
    done_waiting.WaitForNotification();
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
