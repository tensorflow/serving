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

#ifndef TENSORFLOW_SERVING_BATCHING_STREAMING_BATCH_SCHEDULER_H_
#define TENSORFLOW_SERVING_BATCHING_STREAMING_BATCH_SCHEDULER_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/contrib/batching/batch_scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/batching/batch_scheduler_retrier.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {
namespace internal {
class SingleTaskScheduler;
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

namespace tensorflow {
namespace serving {

// A BatchScheduler implementation geared toward handling a single request type
// running on a specific set of hardware resources. A typical scenario is one in
// which all requests invoke the same machine-learned model on one GPU. The
// scheduler streams requests (tasks) to the batch thread while a given batch
// is still filling up, giving the option to process them in a streaming fashion
// in the request thread and/or the batch thread.
//
//
// PARAMETERS AND BEHAVIOR:
//
// StreamingBatchScheduler is parameterized by a maximum batch size and timeout.
// It constructs batches one at a time, stopping when one of these conditions
// occurs:
//  (a) the next task would cause the batch to exceed the size target;
//  (b) waiting for more tasks to be added would exceed the timeout.
//
// Batches are processed in a fixed-size thread pool. When a new batch is
// started it is immediately assigned to a thread while it is being filled with
// tasks. The process-batch callback running in the thread has the option to
// process the tasks in a "streaming" fashion as they arrive. Eventually, once
// the batch size or timeout has been reached, the batch gets closed and the
// callback should finish processing it and exit.
//
// StreamingBatchScheduler does not enqueue tasks if the threads are all busy.
// Every task is either immediately added to a batch that is being serviced by
// an active thread, or rejected with an UNAVAILABLE error (the client may
// subsequently retry submitting the task).
//
//
// RECOMMENDED USE-CASES:
//
// Please see the RECOMMENDED USE-CASES section of BasicBatchScheduler's class
// documentation. The same applies here.
//
//
// EXAMPLE USE-CASE FLOW:
//
// For such use-cases, request processing via StreamingBatchScheduler generally
// follows this flow (given for illustration; variations are possible):
//  1. Optionally perform some pre-processing on each request in the request
//     threads.
//  2. Route the requests to the batch scheduler, as batching::Task objects.
//     (Since all requests are of the same type and are not versioned, the
//     scheduler is free to group them into batches arbitrarily.)
//  3. Optionally perform some pre-processing on the requests in the batching
//     thread as a given batch fills up, perhaps including starting to merge the
//     requests into their single batched representation.
//  4. Wait for the batch to be closed, e.g. by calling WaitUntilClosed(). (Note
//     that the batch will become closed automatically, based on reaching either
//     the maximum batch size or the timeout.)
//  5. Merge the requests into a single batched representation B.
//  6. Obtain handles to the servable(s) needed to process B. The simplest
//     approach is to obtain the latest version of each servable. Alternatively,
//     if cross-servable consistency is required (e.g. the vocabulary lookup
//     table's version number must match that of the tensorflow session),
//     identify an appropriate version number and obtain the servable handles
//     accordingly.
//  7. Process B using the obtained servable handles, and split the result into
//     individual per-request units.
//  8. Perform any post-processing in the batch thread and/or request thread.
//
template <typename TaskType>
class StreamingBatchScheduler : public BatchScheduler<TaskType> {
 public:
  // TODO(b/25089730): Tune defaults based on best practices as they develop.
  struct Options {
    constexpr Options() {}

    // The maximum size of each batch.
    //
    // The scheduler may form batches of any size between 1 and this number
    // (inclusive). If there is a need to quantize the batch sizes, i.e. only
    // submit batches whose size is in a small set of allowed sizes, that can be
    // done by adding padding in the process-batch callback.
    size_t max_batch_size = 1000;

    // The maximum amount of time a task can sit in a batch before the scheduler
    // closes the batch, in microseconds.
    //
    // Setting this value to 0 will *not* result in the behavior of processing
    // a batch as soon as a thread becomes available. Instead, it will cause
    // each batch to contain just a single item, essentially disabling batching.
    // StreamingBatchScheduler is not the right vehicle for achieving the
    // aforementioned behavior.
    //
    // A negative value means that no timeout will be enforced. This setting is
    // useful in some test code.
    int64 batch_timeout_micros = 0;

    // The name to use for the pool of batch threads.
    string thread_pool_name = "batch_threads";

    // The number of threads to use to process batches.
    // Must be >= 1, and should be tuned carefully.
    int num_batch_threads = port::NumSchedulableCPUs();

    // The following options are typically only overridden by test code.

    // The environment to use.
    Env* env = Env::Default();

    // How long SingleTaskScheduler should wait if there are no scheduled tasks,
    // in microseconds.
    uint64 no_tasks_wait_time_micros = 1000;  // 1 millisecond
  };
  static Status Create(
      const Options& options,
      std::function<void(std::unique_ptr<Batch<TaskType>>)>
          process_batch_callback,
      std::unique_ptr<StreamingBatchScheduler<TaskType>>* scheduler);

  ~StreamingBatchScheduler() override;

  Status Schedule(std::unique_ptr<TaskType>* task) override;

  // StreamingBatchScheduler never enqueues tasks, as discussed above.
  size_t NumEnqueuedTasks() const override { return 0; }

  // Scheduling capacity is based purely on threads that can accept tasks
  // immediately (there is no queueing).
  size_t SchedulingCapacity() const override;

 private:
  StreamingBatchScheduler(const Options& options,
                          std::function<void(std::unique_ptr<Batch<TaskType>>)>
                              process_batch_callback);

  // Determines whether it is legal to add 'task' to 'batch'.
  bool TaskFitsInBatch(const TaskType* task,
                       const Batch<TaskType>* batch) const;

  // Closes 'open_batch_' (unless it equals nullptr), and replaces it with a
  // fresh open batch. Schedules the new batch on 'batch_threads_'.
  void StartNewBatch() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Takes a snapshot of 'open_batch_num_', and schedules an event with
  // 'batch_closer_' to close it at time 'close_time_micros' if it is still open
  // at that time.
  void ScheduleCloseOfCurrentOpenBatch(uint64 close_time_micros)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Options options_;

  // A callback invoked to processes a batch of work units. Always invoked from
  // a batch thread.
  std::function<void(std::unique_ptr<Batch<TaskType>>)> process_batch_callback_;

  // A pool of 'options_.num_batch_threads' batch threads.
  std::unique_ptr<thread::ThreadPool> batch_threads_;

  // A mutex protecting 'open_batch_' and associated metadata.
  mutable mutex mu_;

  // The batch that is currently open and into which new tasks can be added.
  // Not owned here; owned by the batch thread pool.
  Batch<TaskType>* open_batch_ GUARDED_BY(mu_) = nullptr;

  // The sequence number of 'open_batch_'. Incremented each time 'open_batch_'
  // is assigned to a new (non-null) batch object.
  int64 open_batch_num_ GUARDED_BY(mu_) = 0;

  // The number of batches "in progress", i.e. batches that have been started
  // but for which the process-batch callback hasn't finished. Note that this
  // counter is somewhat conservative (i.e. might be an overestimate), because
  // it gets decremented after the callback finishes and there could be races.
  int num_batches_in_progress_ GUARDED_BY(mu_) = 0;

  // A background task we use to schedule batches to close when they hit their
  // timeout.
  std::unique_ptr<internal::SingleTaskScheduler> batch_closer_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(StreamingBatchScheduler);
};

// Constructs a StreamingBatchScheduler wrapped with a retrier, for convenience.
template <typename TaskType>
Status CreateRetryingStreamingBatchScheduler(
    const typename StreamingBatchScheduler<TaskType>::Options& schedule_options,
    const typename BatchSchedulerRetrier<TaskType>::Options& retry_options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* scheduler);

//////////
// Implementation details follow. API users need not read.

namespace internal {

// A way to defer a computation until a specific time in the future.
// Spawns a background thread that sleeps and then runs the computation.
// While the computation is waiting to run, the caller may update the time and/
// or computation that gets run. The new update supercedes the old one.
class SingleTaskScheduler {
 public:
  SingleTaskScheduler(Env* env, string thread_name,
                      uint64 no_tasks_wait_time_micros);

  // Blocks until the currently-set closure (if any) runs.
  ~SingleTaskScheduler();

  // Schedules 'closure' to run at time 'time_micros' (in env_ time units). May
  // be called zero or more times. Each call supercedes any prior calls, and
  // cancels any closures provided in them (if they haven't already been run).
  //
  // IMPORTANT: 'time_micros' must be monotonically non-decreasing across calls.
  void Schedule(uint64 time_micros, std::function<void()> closure);

 private:
  // The code executed in 'thread_'. Looks for updated tasks, and executes them
  // by sleeping for the requisite time and then (if no intervening tasks have
  // come in) invoking the callback. Loops until 'stop_' has been notified.
  void ThreadLogic();

  // The environment to use.
  Env* env_;

  mutable mutex mu_;

  // The arguments to Schedule().
  struct Task {
    uint64 time_micros;
    std::function<void()> closure;
  };

  // A newly-scheduled task hasn't yet been picked up by 'thread_'.
  optional<Task> updated_task_ GUARDED_BY(mu_);

  // The time parameter passed in the most recent Schedule() invocation.
  // Used to enforce monotonicity.
  uint64 last_task_time_ = 0;

  // A notification for stopping the thread, during destruction.
  Notification stop_;

  // The name of 'thread_'.
  const string thread_name_;

  // A background thread that runs closures supplied via Schedule().
  std::unique_ptr<Thread> thread_;

  // How long to wait if there are no scheduled tasks, in microseconds.
  const uint64 no_tasks_wait_time_micros_;

  TF_DISALLOW_COPY_AND_ASSIGN(SingleTaskScheduler);
};

}  // namespace internal

template <typename TaskType>
Status StreamingBatchScheduler<TaskType>::Create(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<StreamingBatchScheduler<TaskType>>* scheduler) {
  if (options.max_batch_size <= 0) {
    return errors::InvalidArgument("max_batch_size must be positive; was ",
                                   options.max_batch_size);
  }
  if (options.num_batch_threads <= 0) {
    return errors::InvalidArgument("num_batch_threads must be positive; was ",
                                   options.num_batch_threads);
  }
  scheduler->reset(
      new StreamingBatchScheduler<TaskType>(options, process_batch_callback));
  return Status::OK();
}

template <typename TaskType>
StreamingBatchScheduler<TaskType>::~StreamingBatchScheduler() {
  {
    mutex_lock l(mu_);
    if (open_batch_ != nullptr) {
      open_batch_->Close();
      open_batch_ = nullptr;
      ++open_batch_num_;
    }
  }
  // The thread pool destructor will block until the threads have finished
  // processing the batches.
  batch_threads_.reset(nullptr);
}

template <typename TaskType>
Status StreamingBatchScheduler<TaskType>::Schedule(
    std::unique_ptr<TaskType>* task) {
  if ((*task)->size() > options_.max_batch_size) {
    return errors::InvalidArgument("Task size ", (*task)->size(),
                                   " is larger than maximum batch size ",
                                   options_.max_batch_size);
  }

  {
    mutex_lock l(mu_);

    if (open_batch_ == nullptr || !TaskFitsInBatch(task->get(), open_batch_)) {
      StartNewBatch();
    }

    // Given N threads, if there are N+1 batches then the N+1st batch is empty
    // and is waiting to be assigned a thread. In that situation we reject new
    // tasks with a transient UNAVAILABLE error code.
    if (num_batches_in_progress_ > options_.num_batch_threads) {
      DCHECK(open_batch_->empty());
      return errors::Unavailable(
          "This task would start a fresh batch, but all batch threads are "
          "busy, so at present there is no processing capacity available for "
          "this task");
    }

    // If we are about to add the first task to a batch, schedule the batch to
    // be closed after the timeout.
    if (options_.batch_timeout_micros > 0 && open_batch_->empty()) {
      const uint64 batch_deadline =
          options_.env->NowMicros() + options_.batch_timeout_micros;
      ScheduleCloseOfCurrentOpenBatch(batch_deadline);
    }

    open_batch_->AddTask(std::move(*task));

    // If we've exactly reached the target size, we can close this batch now.
    if (open_batch_->size() == options_.max_batch_size) {
      StartNewBatch();
    }
  }

  return Status::OK();
}

template <typename TaskType>
size_t StreamingBatchScheduler<TaskType>::SchedulingCapacity() const {
  mutex_lock l(mu_);
  if (num_batches_in_progress_ > options_.num_batch_threads) {
    return 0;
  }
  const int num_idle_threads =
      options_.num_batch_threads - num_batches_in_progress_;
  const int open_batch_capacity =
      open_batch_ == nullptr ? 0
                             : options_.max_batch_size - open_batch_->size();
  return (num_idle_threads * options_.max_batch_size) + open_batch_capacity;
}

template <typename TaskType>
StreamingBatchScheduler<TaskType>::StreamingBatchScheduler(
    const Options& options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback)
    : options_(options),
      process_batch_callback_(process_batch_callback),
      batch_threads_(new thread::ThreadPool(options_.env,
                                            options_.thread_pool_name,
                                            options_.num_batch_threads)) {}

template <typename TaskType>
bool StreamingBatchScheduler<TaskType>::TaskFitsInBatch(
    const TaskType* task, const Batch<TaskType>* batch) const {
  return batch->size() + task->size() <= options_.max_batch_size;
}

template <typename TaskType>
void StreamingBatchScheduler<TaskType>::StartNewBatch() {
  if (open_batch_ != nullptr) {
    open_batch_->Close();
    open_batch_ = nullptr;
  }

  Batch<TaskType>* new_open_batch = new Batch<TaskType>;
  ++num_batches_in_progress_;  // Critically, increment *outside* the callback.
  batch_threads_->Schedule([this, new_open_batch] {
    this->process_batch_callback_(
        std::unique_ptr<Batch<TaskType>>(new_open_batch));
    {
      mutex_lock l(this->mu_);
      --this->num_batches_in_progress_;
    }
  });
  open_batch_ = new_open_batch;
  ++open_batch_num_;
}

template <typename TaskType>
void StreamingBatchScheduler<TaskType>::ScheduleCloseOfCurrentOpenBatch(
    uint64 close_time_micros) {
  if (batch_closer_ == nullptr) {
    batch_closer_.reset(new internal::SingleTaskScheduler(
        options_.env, "batch_closer", options_.no_tasks_wait_time_micros));
  }

  const int64 batch_num_to_close = open_batch_num_;
  batch_closer_->Schedule(close_time_micros, [this, batch_num_to_close] {
    {
      mutex_lock l(this->mu_);
      if (open_batch_num_ == batch_num_to_close) {
        StartNewBatch();
      }
    }
  });
}

template <typename TaskType>
Status CreateRetryingStreamingBatchScheduler(
    const typename StreamingBatchScheduler<TaskType>::Options& schedule_options,
    const typename BatchSchedulerRetrier<TaskType>::Options& retry_options,
    std::function<void(std::unique_ptr<Batch<TaskType>>)>
        process_batch_callback,
    std::unique_ptr<BatchScheduler<TaskType>>* scheduler) {
  std::unique_ptr<StreamingBatchScheduler<TaskType>> streaming_scheduler;
  TF_RETURN_IF_ERROR(StreamingBatchScheduler<TaskType>::Create(
      schedule_options, process_batch_callback, &streaming_scheduler));
  std::unique_ptr<BatchSchedulerRetrier<TaskType>> retrier;
  TF_RETURN_IF_ERROR(BatchSchedulerRetrier<TaskType>::Create(
      retry_options, std::move(streaming_scheduler), &retrier));
  *scheduler = std::move(retrier);
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_STREAMING_BATCH_SCHEDULER_H_
