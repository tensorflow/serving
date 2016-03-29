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

#ifndef TENSORFLOW_SERVING_UTIL_THREADPOOL_EXECUTOR_H_
#define TENSORFLOW_SERVING_UTIL_THREADPOOL_EXECUTOR_H_

#include <functional>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/util/executor.h"

namespace tensorflow {
namespace serving {

// An executor which uses a pool of threads to execute the scheduled closures.
class ThreadPoolExecutor : public Executor {
 public:
  // Constructs a threadpool that has 'num_threads' threads with specified
  // 'thread_pool_name'. Env is used to start the thread.
  //
  // REQUIRES: num_threads > 0.
  ThreadPoolExecutor(Env* env, const string& thread_pool_name, int num_threads);

  // Waits until all scheduled work has finished and then destroy the set of
  // threads.
  ~ThreadPoolExecutor() override;

  void Schedule(std::function<void()> fn) override;

 private:
  thread::ThreadPool thread_pool_;

  TF_DISALLOW_COPY_AND_ASSIGN(ThreadPoolExecutor);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_THREADPOOL_EXECUTOR_H_
