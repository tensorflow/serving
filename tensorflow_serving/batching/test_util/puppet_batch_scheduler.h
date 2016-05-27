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

#ifndef TENSORFLOW_SERVING_BATCHING_TEST_UTIL_PUPPET_BATCH_SCHEDULER_H_
#define TENSORFLOW_SERVING_BATCHING_TEST_UTIL_PUPPET_BATCH_SCHEDULER_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <utility>

#include "tensorflow_serving/batching/batch_scheduler.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A BatchScheduler implementation that enqueues tasks, and when requested via
// a method call places them into a batch and runs the batch. (It doesn't have
// any threads or enforce any maximum batch size or timeout.)
//
// This scheduler is useful for testing classes whose implementation relies on
// a batch scheduler. The class under testing can be configured to use a
// PuppetBatchScheduler, in lieu of a real one, for the purpose of the test.
template <typename TaskType>
class PuppetBatchScheduler : public BatchScheduler<TaskType> {
 public:
  explicit PuppetBatchScheduler(
      std::function<void(std::unique_ptr<Batch<TaskType>>)>
          process_batch_callback);
  ~PuppetBatchScheduler() override = default;

  Status Schedule(std::unique_ptr<TaskType>* task) override;

  size_t NumEnqueuedTasks() const override;

  // This schedule has unbounded capacity, so this method returns the maximum
  // size_t value to simulate infinity.
  size_t SchedulingCapacity() const override;

  // Processes up to 'num_tasks' enqueued tasks, in FIFO order.
  void ProcessTasks(int num_tasks);

  // Processes all enqueued tasks.
  void ProcessAllTasks();

 private:
  std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback_;

  // Tasks submitted to the scheduler. Processed in FIFO order.
  std::queue<std::unique_ptr<TaskType>> queue_;

  TF_DISALLOW_COPY_AND_ASSIGN(PuppetBatchScheduler);
};

//////////
// Implementation details follow. API users need not read.

template <typename TaskType>
PuppetBatchScheduler<TaskType>::PuppetBatchScheduler(
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback)
    : process_batch_callback_(process_batch_callback) {}

template <typename TaskType>
Status PuppetBatchScheduler<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  queue_.push(std::move(*task));
  return Status::OK();
}

template <typename TaskType>
size_t PuppetBatchScheduler<TaskType>::NumEnqueuedTasks() const {
  return queue_.size();
}

template <typename TaskType>
size_t PuppetBatchScheduler<TaskType>::SchedulingCapacity() const {
  return std::numeric_limits<size_t>::max();
}

template <typename TaskType>
void PuppetBatchScheduler<TaskType>::ProcessTasks(int num_tasks) {
  if (queue_.empty()) {
    return;
  }
  auto batch = std::unique_ptr<Batch<TaskType>>(new Batch<TaskType>);
  while (batch->num_tasks() < num_tasks && !queue_.empty()) {
    batch->AddTask(std::move(queue_.front()));
    queue_.pop();
  }
  batch->Close();
  process_batch_callback_(std::move(batch));
}

template <typename TaskType>
void PuppetBatchScheduler<TaskType>::ProcessAllTasks() {
  ProcessTasks(queue_.size());
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_TEST_UTIL_PUPPET_BATCH_SCHEDULER_H_
