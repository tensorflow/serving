/* Copyright 2016 IBM Corp. All Rights Reserved. */
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
