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

#ifndef TENSORFLOW_SERVING_UTIL_SHARED_COUNTER_H_
#define TENSORFLOW_SERVING_UTIL_SHARED_COUNTER_H_

#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace serving {

// A thread-safe counter that supports increment, decrement, and waiting until
// it gets incremented/decremented.
class SharedCounter {
 public:
  // The counter value starts at 0.
  SharedCounter() = default;
  ~SharedCounter() = default;

  // Reads the current counter value.
  int GetValue() const;

  // Increments the counter by 1.
  void Increment();

  // Decrements the counter by 1.
  void Decrement();

  // Blocks until the counter value changes (via Increment() or Decrement()).
  void WaitUntilChanged();

 private:
  // Notifies all entries in 'pending_notifications_', and then clears it.
  void NotifyAndClearAllPendingNotifications() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;

  // The current counter value.
  int count_ GUARDED_BY(mu_) = 0;

  // Notifications that are waiting for 'count_' to be updated.
  std::vector<Notification*> pending_notifications_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(SharedCounter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_SHARED_COUNTER_H_
