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

#include "tensorflow_serving/util/shared_counter.h"

namespace tensorflow {
namespace serving {

int SharedCounter::GetValue() const {
  mutex_lock l(mu_);
  return count_;
}

void SharedCounter::Increment() {
  mutex_lock l(mu_);
  ++count_;
  NotifyAndClearAllPendingNotifications();
}

void SharedCounter::Decrement() {
  mutex_lock l(mu_);
  --count_;
  NotifyAndClearAllPendingNotifications();
}

void SharedCounter::WaitUntilChanged() {
  Notification notification;
  {
    mutex_lock l(mu_);
    pending_notifications_.push_back(&notification);
  }
  notification.WaitForNotification();
}

void SharedCounter::NotifyAndClearAllPendingNotifications() {
  for (Notification* notification : pending_notifications_) {
    notification->Notify();
  }
  pending_notifications_.clear();
}

}  // namespace serving
}  // namespace tensorflow
