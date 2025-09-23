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

#include <functional>
#include <utility>

#include "absl/types/optional.h"

namespace tensorflow {
namespace serving {

namespace internal {

// SingleTaskScheduler

SingleTaskScheduler::SingleTaskScheduler(Env* env, string thread_name,
                                         uint64_t no_tasks_wait_time_micros)
    : env_(env),
      thread_name_(std::move(thread_name)),
      no_tasks_wait_time_micros_(no_tasks_wait_time_micros) {}

SingleTaskScheduler::~SingleTaskScheduler() { stop_.Notify(); }

void SingleTaskScheduler::Schedule(uint64_t time_micros,
                                   std::function<void()> closure) {
  DCHECK_GE(time_micros, last_task_time_);
  last_task_time_ = time_micros;

  {
    mutex_lock l(mu_);
    updated_task_ = {time_micros, std::move(closure)};
  }

  if (thread_ == nullptr) {
    ThreadOptions options;
    thread_.reset(env_->StartThread(options, thread_name_,
                                    [this] { this->ThreadLogic(); }));
  }
}

void SingleTaskScheduler::ThreadLogic() {
  absl::optional<Task> current_task = absl::nullopt;
  for (;;) {
    // Sleep until the time specified in the current task, if any.
    if (current_task) {
      const uint64_t now = env_->NowMicros();
      if (current_task->time_micros > now) {
        env_->SleepForMicroseconds(current_task->time_micros - now);
      }
    }

    // Check for an updated task.
    {
      mutex_lock l(mu_);
      if (updated_task_) {
        current_task = updated_task_;
        updated_task_ = absl::nullopt;
        // We've got an updated task. Start over.
        continue;
      }
    }

    // Invoke the closure of the current task, if any. Otherwise, we've got
    // nothing to do, so sleep for a spell.
    if (current_task) {
      current_task->closure();
      current_task = absl::nullopt;
    } else {
      if (stop_.HasBeenNotified()) {
        return;
      }
      env_->SleepForMicroseconds(no_tasks_wait_time_micros_);
    }
  }
}

}  // namespace internal

}  // namespace serving
}  // namespace tensorflow
