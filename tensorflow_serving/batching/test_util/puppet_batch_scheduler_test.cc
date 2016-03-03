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

#include "tensorflow_serving/batching/test_util/puppet_batch_scheduler.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace serving {
namespace test_util {
namespace {

class FakeTask : public BatchTask {
 public:
  FakeTask() = default;
  ~FakeTask() override = default;

  size_t size() const override { return 1; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

// Creates a FakeTask and calls 'scheduler->Schedule()' on that task, and
// expects the call to succeed.
void ScheduleTask(BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask);
  TF_ASSERT_OK(scheduler->Schedule(&task));
}

TEST(PuppetBatchSchedulerTest, Basic) {
  int num_tasks_processed = 0;
  auto callback =
      [&num_tasks_processed](std::unique_ptr<Batch<FakeTask>> batch) {
        ASSERT_TRUE(batch->IsClosed());
        num_tasks_processed += batch->size();
      };
  PuppetBatchScheduler<FakeTask> scheduler(callback);

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(0, num_tasks_processed);
    EXPECT_EQ(i, scheduler.NumEnqueuedTasks());
    ScheduleTask(&scheduler);
  }
  EXPECT_EQ(3, scheduler.NumEnqueuedTasks());

  scheduler.ProcessTasks(2);
  EXPECT_EQ(2, num_tasks_processed);
  EXPECT_EQ(1, scheduler.NumEnqueuedTasks());

  ScheduleTask(&scheduler);
  EXPECT_EQ(2, num_tasks_processed);
  EXPECT_EQ(2, scheduler.NumEnqueuedTasks());

  scheduler.ProcessAllTasks();
  EXPECT_EQ(4, num_tasks_processed);
  EXPECT_EQ(0, scheduler.NumEnqueuedTasks());
}

}  // namespace
}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
