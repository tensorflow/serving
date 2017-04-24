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

#include "tensorflow_serving/batching/batch_scheduler_retrier.h"

#include <limits>

#include <gtest/gtest.h>
#include "tensorflow/contrib/batching/test_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace serving {
namespace {

class FakeTask : public BatchTask {
 public:
  FakeTask() = default;
  ~FakeTask() override = default;

  size_t size() const override { return 1; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

// A batch scheduler that always fails with an UNKNOWN status.
class BrokenScheduler : public BatchScheduler<FakeTask> {
 public:
  BrokenScheduler() = default;
  ~BrokenScheduler() override = default;

  Status Schedule(std::unique_ptr<FakeTask>* task) override {
    ++num_submit_calls_;
    return errors::Unknown("BrokenScheduler faithfully failing");
  }

  size_t NumEnqueuedTasks() const override { return 7; }

  size_t SchedulingCapacity() const override { return 42; }

  int num_submit_calls() const { return num_submit_calls_; }

 private:
  int num_submit_calls_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(BrokenScheduler);
};

// A batch scheduler that fails with an UNAVAILABLE status the first N-1 times
// and then succeeds.
class StubbornScheduler : public BatchScheduler<FakeTask> {
 public:
  explicit StubbornScheduler(int num_attempts_to_succeed)
      : num_attempts_to_succeed_(num_attempts_to_succeed) {}
  ~StubbornScheduler() override = default;

  Status Schedule(std::unique_ptr<FakeTask>* task) override {
    ++num_attempts_;
    if (num_attempts_ >= num_attempts_to_succeed_) {
      std::unique_ptr<FakeTask> consumed_task = std::move(*task);
      return Status::OK();
    } else {
      return errors::Unavailable(
          "StubbornScheduler faithfully being stubborn; this is attempt ",
          num_attempts_);
    }
  }

  size_t NumEnqueuedTasks() const override { return 0; }

  size_t SchedulingCapacity() const override {
    return std::numeric_limits<size_t>::max();
  }

  int num_attempts() const { return num_attempts_; }

 private:
  const int num_attempts_to_succeed_;
  int num_attempts_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(StubbornScheduler);
};

TEST(BatchSchedulerRetrierTest, ConstMethodsForwardToWrappedScheduler) {
  auto broken_scheduler = std::unique_ptr<BrokenScheduler>(new BrokenScheduler);
  BatchSchedulerRetrier<FakeTask>::Options options;
  std::unique_ptr<BatchSchedulerRetrier<FakeTask>> retrier;
  TF_CHECK_OK(BatchSchedulerRetrier<FakeTask>::Create(
      options, std::move(broken_scheduler), &retrier));
  EXPECT_EQ(7, retrier->NumEnqueuedTasks());
  EXPECT_EQ(42, retrier->SchedulingCapacity());
}

TEST(BatchSchedulerRetrierTest, PermanentFailure) {
  auto broken_scheduler = std::unique_ptr<BrokenScheduler>(new BrokenScheduler);
  auto broken_scheduler_ptr = broken_scheduler.get();
  BatchSchedulerRetrier<FakeTask>::Options options;
  std::unique_ptr<BatchSchedulerRetrier<FakeTask>> retrier;
  TF_CHECK_OK(BatchSchedulerRetrier<FakeTask>::Create(
      options, std::move(broken_scheduler), &retrier));
  auto task = std::unique_ptr<FakeTask>(new FakeTask);
  Status status = retrier->Schedule(&task);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(error::UNKNOWN, status.code());
  EXPECT_FALSE(task == nullptr);
  EXPECT_EQ(1, broken_scheduler_ptr->num_submit_calls());
}

TEST(BatchSchedulerRetrierTest, MaxTime) {
  for (int num_attempts_to_succeed = 1; num_attempts_to_succeed < 3;
       ++num_attempts_to_succeed) {
    for (int max_attempts = 1; max_attempts < 5; ++max_attempts) {
      test_util::FakeClockEnv env(Env::Default());

      auto stubborn_scheduler = std::unique_ptr<StubbornScheduler>(
          new StubbornScheduler(num_attempts_to_succeed));
      auto stubborn_scheduler_ptr = stubborn_scheduler.get();
      BatchSchedulerRetrier<FakeTask>::Options options;
      options.retry_delay_micros = 1;
      options.max_time_micros = max_attempts;
      options.env = &env;
      std::unique_ptr<BatchSchedulerRetrier<FakeTask>> retrier;
      TF_CHECK_OK(BatchSchedulerRetrier<FakeTask>::Create(
          options, std::move(stubborn_scheduler), &retrier));

      const bool expect_success = max_attempts >= num_attempts_to_succeed;
      Notification done;
      std::unique_ptr<Thread> run_retrier(Env::Default()->StartThread(
          {}, "RunRetrier",
          [&retrier, &expect_success, &done]() {
            auto task = std::unique_ptr<FakeTask>(new FakeTask);
            Status status = retrier->Schedule(&task);
            EXPECT_EQ(expect_success, status.ok());
            if (!status.ok()) {
              EXPECT_EQ(error::UNAVAILABLE, status.code());
            }
            EXPECT_EQ(expect_success, task == nullptr);
            done.Notify();
          }));

      for (int attempt = 0; attempt < max_attempts - 1; ++attempt) {
        if (attempt >= num_attempts_to_succeed - 1) {
          break;
        }
        env.BlockUntilThreadsAsleep(1);
        EXPECT_EQ(attempt + 1, stubborn_scheduler_ptr->num_attempts());
        env.AdvanceByMicroseconds(options.retry_delay_micros);
      }
      done.WaitForNotification();
    }
  }
}

TEST(BatchSchedulerRetrierTest, RetryDelay) {
  test_util::FakeClockEnv env(Env::Default());

  const int num_attempts_to_succeed = 3;
  auto stubborn_scheduler = std::unique_ptr<StubbornScheduler>(
      new StubbornScheduler(num_attempts_to_succeed));
  auto stubborn_scheduler_ptr = stubborn_scheduler.get();
  BatchSchedulerRetrier<FakeTask>::Options options;
  options.retry_delay_micros = 7;
  options.max_time_micros = 100;
  options.env = &env;
  std::unique_ptr<BatchSchedulerRetrier<FakeTask>> retrier;
  TF_CHECK_OK(BatchSchedulerRetrier<FakeTask>::Create(
      options, std::move(stubborn_scheduler), &retrier));

  Notification done;
  std::unique_ptr<Thread> run_retrier(Env::Default()->StartThread(
      {}, "RunRetrier",
      [&retrier, &done]() {
        auto task = std::unique_ptr<FakeTask>(new FakeTask);
        Status status = retrier->Schedule(&task);
        TF_EXPECT_OK(status);
        done.Notify();
      }));

  for (int attempt = 0; attempt < num_attempts_to_succeed - 1; ++attempt) {
    env.BlockUntilThreadsAsleep(1);
    EXPECT_EQ(attempt + 1, stubborn_scheduler_ptr->num_attempts());
    env.AdvanceByMicroseconds(options.retry_delay_micros);
  }
  done.WaitForNotification();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
