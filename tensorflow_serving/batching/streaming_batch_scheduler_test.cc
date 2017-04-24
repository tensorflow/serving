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

#include "tensorflow_serving/batching/streaming_batch_scheduler.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/batching/test_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

namespace tensorflow {
namespace serving {
namespace {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

 private:
  const size_t size_;

  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

// Creates a FakeTask of size 'task_size', and calls 'scheduler->Schedule()' on
// that task. Returns the resulting status.
Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

TEST(StreamingBatchSchedulerTest, Basic) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    callback_called = true;
    batch->WaitUntilClosed();
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());
    EXPECT_EQ(5, batch->task(1).size());
  };
  {
    StreamingBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
    options.num_batch_threads = 1;
    std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(StreamingBatchScheduler<FakeTask>::Create(options, callback,
                                                           &scheduler));
    TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
    TF_ASSERT_OK(ScheduleTask(5, scheduler.get()));
  }
  EXPECT_TRUE(callback_called);
}

TEST(StreamingBatchSchedulerTest, ObeyBatchSizeConstraint) {
  // Set up a callback that captures the batches' task sizes.
  mutex mu;
  std::vector<std::vector<size_t>> callback_data;
  auto callback = [&mu,
                   &callback_data](std::unique_ptr<Batch<FakeTask>> batch) {
    batch->WaitUntilClosed();
    std::vector<size_t> batch_data;
    for (int i = 0; i < batch->num_tasks(); ++i) {
      batch_data.push_back(batch->mutable_task(i)->size());
    }
    {
      mutex_lock l(mu);
      callback_data.push_back(batch_data);
    }
  };

  // Run a batch scheduler and inject some tasks.
  {
    StreamingBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
    options.num_batch_threads = 2;
    std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(StreamingBatchScheduler<FakeTask>::Create(options, callback,
                                                           &scheduler));

    // First batch.
    TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
    TF_ASSERT_OK(ScheduleTask(5, scheduler.get()));

    // Second batch (due to size overage).
    TF_ASSERT_OK(ScheduleTask(3 /* (3+5) + 3 > 10 */, scheduler.get()));
    TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));
    TF_ASSERT_OK(ScheduleTask(6, scheduler.get()));

    // (Empty third batch, since the second batch exactly hit the size limit.)
  }

  // Expect a certain grouping of the tasks into batches.
  EXPECT_THAT(
      callback_data,
      UnorderedElementsAre(ElementsAre(3, 5), ElementsAre(3, 1, 6), IsEmpty()));
}

TEST(StreamingBatchSchedulerTest, Timeout) {
  // Set up a fake clock, which only advances when we explicitly tell it to.
  test_util::FakeClockEnv env(Env::Default());

  Notification first_batch_processed, second_batch_processed,
      third_batch_processed;
  auto callback = [&first_batch_processed, &second_batch_processed,
                   &third_batch_processed](
      std::unique_ptr<Batch<FakeTask>> batch) {
    batch->WaitUntilClosed();
    if (batch->size() == 1) {
      first_batch_processed.Notify();
    } else if (batch->size() == 2) {
      second_batch_processed.Notify();
    } else if (batch->size() == 3) {
      third_batch_processed.Notify();
    }
  };

  StreamingBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 4;
  options.batch_timeout_micros = 10;
  options.num_batch_threads = 10;  // Plenty of threads to avoid "fullness".
  options.env = &env;
  // Set non-timeout-related sleep times to 0 for this test.
  options.no_tasks_wait_time_micros = 0;
  std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      StreamingBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // Create an underfull batch, and ensure that it gets processed when the clock
  // hits the timeout.
  TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));
  env.BlockUntilSleepingThread(10);
  env.AdvanceByMicroseconds(9);
  Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
  EXPECT_FALSE(first_batch_processed.HasBeenNotified());
  env.AdvanceByMicroseconds(1);
  first_batch_processed.WaitForNotification();

  // Start creating a batch, then advance the clock until just before the
  // timeout. Then submit a new task that overflows into the next batch, causing
  // the original batch to close.
  TF_ASSERT_OK(ScheduleTask(2, scheduler.get()));
  env.BlockUntilSleepingThread(20);
  env.AdvanceByMicroseconds(9);
  Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
  EXPECT_FALSE(second_batch_processed.HasBeenNotified());
  TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
  second_batch_processed.WaitForNotification();

  // Allow the third batch to hit its timeout, and ensure it gets closed at the
  // right time.
  env.AdvanceByMicroseconds(9);
  Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
  EXPECT_FALSE(third_batch_processed.HasBeenNotified());
  env.BlockUntilSleepingThread(29);
  env.AdvanceByMicroseconds(1);
  third_batch_processed.WaitForNotification();
}

TEST(StreamingBatchSchedulerTest, RealClockTimeout) {
  Notification first_batch_processed, second_batch_processed;
  auto callback = [&first_batch_processed, &second_batch_processed](
      std::unique_ptr<Batch<FakeTask>> batch) {
    batch->WaitUntilClosed();
    if (batch->size() == 1) {
      first_batch_processed.Notify();
    } else if (batch->size() == 2) {
      second_batch_processed.Notify();
    }
  };

  StreamingBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 10;
  options.batch_timeout_micros = 10 * 1000;  // 10 milliseconds
  options.num_batch_threads = 10;  // Plenty of threads to avoid "fullness".
  std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      StreamingBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // Submit a single task that doesn't fill up the batch.
  // Ensure that it gets processed due to the timeout.
  TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));
  first_batch_processed.WaitForNotification();

  // Do it again.
  TF_ASSERT_OK(ScheduleTask(2, scheduler.get()));
  second_batch_processed.WaitForNotification();
}

TEST(StreamingBatchSchedulerTest, FinalUnderfullBatchProcessedUponDeletion) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    batch->WaitUntilClosed();
    callback_called = true;
  };

  {
    StreamingBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
    options.num_batch_threads = 1;
    std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(StreamingBatchScheduler<FakeTask>::Create(options, callback,
                                                           &scheduler));

    // Submit a single task that doesn't fill up the batch.
    // Ensure that it gets processed when the destructor is called.
    TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
  }
  EXPECT_TRUE(callback_called);
}

TEST(StreamingBatchSchedulerTest, BatchHandedToCallbackWhenFirstCreated) {
  Notification stop_scheduler;
  auto callback = [&stop_scheduler](std::unique_ptr<Batch<FakeTask>> batch) {
    EXPECT_LE(batch->num_tasks(), 1);
    EXPECT_FALSE(batch->IsClosed());
    stop_scheduler.Notify();
    batch->WaitUntilClosed();
  };

  StreamingBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 100;
  options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
  options.num_batch_threads = 1;
  std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      StreamingBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // Submit a single task of size 1, into a batch with much larger capacity.
  TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));

  stop_scheduler.WaitForNotification();
}

TEST(StreamingBatchSchedulerTest, ConstMethods) {
  for (const int num_threads : {1, 2, 3}) {
    Notification proceed;
    auto callback = [&proceed](std::unique_ptr<Batch<FakeTask>> batch) {
      batch->WaitUntilClosed();
      proceed.WaitForNotification();
    };

    StreamingBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 2;
    options.batch_timeout_micros = 1 * 1000 * 1000;  // Don't trigger.
    options.num_batch_threads = num_threads;
    std::unique_ptr<StreamingBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(StreamingBatchScheduler<FakeTask>::Create(options, callback,
                                                           &scheduler));

    // Submit 'num_threads' full batches, to make the scheduling threads "full".
    // (At all times, the queue length should show as 0, since
    // StreamingBatchScheduler never enqueues tasks.)
    for (int i = 0; i < num_threads; ++i) {
      EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
      EXPECT_EQ((num_threads - i) * 2, scheduler->SchedulingCapacity());
      TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));
      EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
      EXPECT_EQ((num_threads - i) * 2 - 1, scheduler->SchedulingCapacity());
      TF_ASSERT_OK(ScheduleTask(1, scheduler.get()));
    }
    EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(0, scheduler->SchedulingCapacity());

    // Make another Schedule() call while the threads are full, which should
    // yield an UNAVAILABLE error.
    Status status = ScheduleTask(1, scheduler.get());
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(error::UNAVAILABLE, status.code());
    EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(0, scheduler->SchedulingCapacity());

    // Allow the processing to proceed, and wait plenty of time for it to finish
    // and the scheduler to get back to full capacity.
    proceed.Notify();
    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100 milliseconds */);

    // Now, SchedulingCapacity() should show as full and Schedule() should
    // succeed.
    EXPECT_EQ(num_threads * 2, scheduler->SchedulingCapacity());
    TF_EXPECT_OK(ScheduleTask(1, scheduler.get()));
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
