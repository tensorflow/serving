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

#ifndef TENSORFLOW_SERVING_TEST_UTIL_FAKE_CLOCK_ENV_H_
#define TENSORFLOW_SERVING_TEST_UTIL_FAKE_CLOCK_ENV_H_

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// An Env implementation with a fake clock for NowMicros() and
// SleepForMicroseconds(). The clock doesn't advance on its own; it advances via
// an explicit Advance() method.
// All other Env virtual methods pass through to a wrapped Env.
class FakeClockEnv : public EnvWrapper {
 public:
  explicit FakeClockEnv(Env* wrapped);
  ~FakeClockEnv() override = default;

  // Advance the clock by a certain number of microseconds.
  void AdvanceByMicroseconds(int micros);

  // Blocks until there is a sleeping thread that is scheduled to wake up at
  // the given (absolute) time.
  void BlockUntilSleepingThread(uint64 wake_time);

  // Blocks until there are at least num_threads sleeping.
  void BlockUntilThreadsAsleep(int num_threads);

  // Methods that this class implements.
  uint64 NowMicros() override;
  void SleepForMicroseconds(int64 micros) override;

 private:
  mutex mu_;

  uint64 current_time_ GUARDED_BY(mu_) = 0;

  struct SleepingThread {
    uint64 wake_time;
    Notification* wake_notification;
  };
  std::vector<SleepingThread> sleeping_threads_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FakeClockEnv);
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_TEST_UTIL_FAKE_CLOCK_ENV_H_
