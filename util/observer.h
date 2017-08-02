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

#ifndef TENSORFLOW_SERVING_UTIL_OBSERVER_H_
#define TENSORFLOW_SERVING_UTIL_OBSERVER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// An observer calls a std::function whenever a specific event happens. The
// difference between an observer and a plain std::function is that it is safe
// to destroy an observer even while external code may be notifying it.
//
// Example use:
//   void ObserveSomeNotifications() {
//     mutex mu;
//     int num_notifications = 0;
//     Observer<> observer([&](){ mutex_lock l(mu); ++num_notifications; });
//     AsynchronouslySendNotificationsForAWhile(observer.Notifier());
//     SleepForALittleWhile();
//     {
//       mutex_lock l(mu);
//       LOG(INFO) << "Currently " << num_notifications << " notifications";
//     }
//     // Let 'observer' fall out of scope and be deleted. Any future
//     // notifications will be no-ops.
//   }
//
// Note that the current implementation serializes the notification calls, so it
// should not be used with long notifications if the inability to overlap them
// would be critical for performance.
//
// Note that naive use of this class will result in "orphaned", no-op instances
// std::function laying around. If this is a concern please use ObserverList to
// manage large collections of observers.
template <typename... Args>
class Observer {
 public:
  // The type of function that this observer will wrap.
  using Function = std::function<void(Args...)>;

  // Wraps 'f' as an observer.
  explicit Observer(Function f) : impl_(std::make_shared<Impl>(std::move(f))) {}

  // Destruction will cause all notifiers to become no-ops.
  ~Observer() {
    if (impl_ != nullptr) {
      impl_->Orphan();
    }
  }

  // Returns a function that will notify this observer for its lifetime.
  // Becomes a no-op after the observer is destroyed.
  Function Notifier() const {
    auto impl = impl_;
    DCHECK(impl != nullptr);  // An implementation invariant.
    return [impl](Args... args) { impl->Notify(std::forward<Args>(args)...); };
  }

 private:
  template <typename... T>
  friend class ObserverList;

  // The underlying implementation.
  class Impl;

  // The implementation, shared with this object and all notifiers.
  std::shared_ptr<Impl> impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(Observer);
};

// An observer list is essentially a std::vector<Observer>, the key difference
// is that an ObserverList will garbage collect orphaned notifiers.
template <typename... Args>
class ObserverList {
 public:
  // Add an observer to the list.
  void Add(const Observer<Args...>& new_observer) {
    // Try to reuse an existing slot if possible.
    for (auto& observer : observers_) {
      if (observer->IsOrphaned()) {
        observer = new_observer.impl_;
        return;
      }
    }
    observers_.push_back(new_observer.impl_);
  }

  // Notify all observers in the list.
  void Notify(Args... args) {
    for (const auto& observer : observers_) {
      observer->Notify(std::forward<Args>(args)...);
    }
  }

  // Clear all observers from this list.
  void Clear() { observers_.clear(); }

 private:
  // The impls of all observers added to this list.
  std::vector<std::shared_ptr<typename Observer<Args...>::Impl>> observers_;
};

//////////
// Implementation details follow. API users need not read.

template <typename... Args>
class Observer<Args...>::Impl {
 public:
  explicit Impl(Function f) : f_(std::move(f)) {}

  bool IsOrphaned() const {
    mutex_lock lock(mutex_);
    return f_ == nullptr;
  }

  void Orphan() {
    mutex_lock lock(mutex_);
    f_ = nullptr;
  }

  void Notify(Args... args) const {
    mutex_lock lock(mutex_);
    if (f_ != nullptr) {
      f_(std::forward<Args>(args)...);
    }
  }

 private:
  mutable mutex mutex_;
  // The function to call when an observed even occurs.
  Function f_ GUARDED_BY(mutex_);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_OBSERVER_H_
