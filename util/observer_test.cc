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

#include "tensorflow_serving/util/observer.h"

#include <gtest/gtest.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/util/periodic_function.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(ObserverTest, Call) {
  int num_calls = 0;
  Observer<> observer([&]() { num_calls++; });
  observer.Notifier()();
  EXPECT_EQ(1, num_calls);
}

TEST(ObserverTest, CallWithArg) {
  Observer<int> observer([&](int arg) { EXPECT_EQ(1337, arg); });
  observer.Notifier()(1337);
}

TEST(ObserverTest, Orphan) {
  int num_calls = 0;
  std::function<void()> notifier;
  {
    Observer<> observer([&]() { num_calls++; });
    notifier = observer.Notifier();
    EXPECT_EQ(0, num_calls);
    notifier();
    EXPECT_EQ(1, num_calls);
  }
  notifier();
  EXPECT_EQ(1, num_calls);
}

TEST(ObserverTest, ObserverList) {
  int num_calls = 0;
  Observer<> observer([&]() { num_calls++; });
  ObserverList<> observers;
  for (int i = 0; i < 10; ++i) {
    observers.Add(observer);
  }
  observers.Notify();
  EXPECT_EQ(10, num_calls);
}

TEST(ObserverTest, ObserverListWithOrphans) {
  int num_calls = 0;
  ObserverList<> observers;
  for (int i = 0; i < 10; ++i) {
    Observer<> observer([&]() { num_calls++; });
    observers.Add(observer);
  }
  observers.Notify();

  // Everything is an orphan, so no calls.
  EXPECT_EQ(0, num_calls);
}

TEST(ObserverTest, Threaded) {
  mutex mu;
  int num_calls = 0;
  auto observer =
      std::unique_ptr<Observer<>>(new Observer<>([&mu, &num_calls]() {
        mutex_lock l(mu);
        ++num_calls;
      }));
  auto notifier = observer->Notifier();

  // Spawn a thread and wait for it to run a few times.
  PeriodicFunction thread(notifier, 1000 /* 1 milliseconds */);
  while (true) {
    {
      mutex_lock l(mu);
      if (num_calls >= 10) {
        break;
      }
    }
    Env::Default()->SleepForMicroseconds(1000 /* 1 milliseconds */);
  }

  // Tear down the observer and make sure it is never called again.
  observer = nullptr;
  int num_calls_snapshot;
  {
    mutex_lock l(mu);
    num_calls_snapshot = num_calls;
  }
  Env::Default()->SleepForMicroseconds(100 * 1000 /* 100 milliseconds */);
  {
    mutex_lock l(mu);
    EXPECT_EQ(num_calls_snapshot, num_calls);
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
