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

#ifndef TENSORFLOW_SERVING_BATCHING_BATCH_SCHEDULER_RETRIER_H_
#define TENSORFLOW_SERVING_BATCHING_BATCH_SCHEDULER_RETRIER_H_

#include <stddef.h>
#include <cstddef>
#include <memory>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/batching/batch_scheduler.h"

namespace tensorflow {
namespace serving {

// A wrapper around another BatchScheduler that automatically retries
// Schedule() requests. Returns an UNAVAILABLE error only after retry attempts
// have failed (based on parameters that govern the maximum number of retries
// and the retry time interval).
template <typename TaskType>
class BatchSchedulerRetrier : public BatchScheduler<TaskType> {
 public:
  struct Options {
    // The maximum amount of time to spend retrying 'wrapped_->Schedule()'
    // calls, in microseconds.
    int64 max_time_micros = 10 * 1000 /* 10 milliseconds */;

    // The amount of time to pause between retry attempts, in microseconds.
    int64 retry_delay_micros = 100;

    // The environment to use for time and sleeping.
    Env* env = Env::Default();
  };
  static Status Create(
      const Options& options, std::unique_ptr<BatchScheduler<TaskType>> wrapped,
      std::unique_ptr<BatchSchedulerRetrier<TaskType>>* result);

  ~BatchSchedulerRetrier() override = default;

  Status Schedule(std::unique_ptr<TaskType>* task) override;
  size_t NumEnqueuedTasks() const override;
  size_t SchedulingCapacity() const override;

 private:
  BatchSchedulerRetrier(const Options& options,
                        std::unique_ptr<BatchScheduler<TaskType>> wrapped);

  const Options options_;
  std::unique_ptr<BatchScheduler<TaskType>> wrapped_;

  TF_DISALLOW_COPY_AND_ASSIGN(BatchSchedulerRetrier);
};

//////////
// Implementation details follow. API users need not read.

template <typename TaskType>
Status BatchSchedulerRetrier<TaskType>::Create(
    const Options& options, std::unique_ptr<BatchScheduler<TaskType>> wrapped,
    std::unique_ptr<BatchSchedulerRetrier<TaskType>>* result) {
  if (options.max_time_micros < 0) {
    return errors::InvalidArgument("max_time_micros must be non-negative; was ",
                                   options.max_time_micros);
  }
  if (options.retry_delay_micros < 0) {
    return errors::InvalidArgument(
        "retry_delay_micros must be non-negative; was ",
        options.retry_delay_micros);
  }
  result->reset(new BatchSchedulerRetrier(options, std::move(wrapped)));
  return Status::OK();
}

template <typename TaskType>
Status BatchSchedulerRetrier<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  Status status;

  const uint64 start_time_micros = options_.env->NowMicros();
  for (;;) {
    status = wrapped_->Schedule(task);
    if (status.code() != error::UNAVAILABLE) {
      // We either succeeded, or got a permanent (non-retriable) error.
      break;
    }
    if ((options_.env->NowMicros() + options_.retry_delay_micros) -
            start_time_micros >=
        options_.max_time_micros) {
      // We don't have time in our budget to retry again.
      break;
    }

    options_.env->SleepForMicroseconds(options_.retry_delay_micros);
  }

  return status;
}

template <typename TaskType>
size_t BatchSchedulerRetrier<TaskType>::NumEnqueuedTasks() const {
  return wrapped_->NumEnqueuedTasks();
}

template <typename TaskType>
size_t BatchSchedulerRetrier<TaskType>::SchedulingCapacity() const {
  return wrapped_->SchedulingCapacity();
}

template <typename TaskType>
BatchSchedulerRetrier<TaskType>::BatchSchedulerRetrier(
    const Options& options, std::unique_ptr<BatchScheduler<TaskType>> wrapped)
    : options_(options), wrapped_(std::move(wrapped)) {}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_BATCH_SCHEDULER_RETRIER_H_
