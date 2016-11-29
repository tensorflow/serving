/*

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

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

#include "simple_thread_sink.h"

// the constructor just launches some amount of workers
SimpleThreadSink::SimpleThreadSink() : stop_(false) {
  worker_ = std::thread([this] {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(this->queue_mutex_);
        this->condition_.wait(
            lock, [this] { return this->stop_ || !this->tasks_.empty(); });

        if (this->stop_ && this->tasks_.empty()) return;

        task = std::move(this->tasks_.front());
        this->tasks_.pop();
      }
      task();
    }
  });
}

// stop and join the worker thread
SimpleThreadSink::~SimpleThreadSink() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
    condition_.notify_all();
  }
  worker_.join();
}
