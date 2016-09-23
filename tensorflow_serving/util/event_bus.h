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

// EventBus enables basic publish / subscribe semantics for events amongst one
// or more publishers and subscribers. The purpose of EventBus for serving is
// to de-couple the code for such events, and is optimized for that use-case.
//
// EventBus is thread-safe. However, if any subscriber callback calls any method
// in the EventBus, it will deadlock.
//
// EventBus and Subscriptions can be destroyed safely in any order. There is a
// strict requirement for memory safety that a Subscription must be destroyed
// before any of the objects or memory that a subscriber's callback accesses.
//
// Important scaling and threading limitations:
//
// Scaling:
// The EventBus is not currently optimized for high scale, either in the number
// of subscribers or frequency of events. For such use-cases, consider alternate
// implementations or upgrades to this class.
//
// Threading:
// Subscribers are notified serially on the event publisher's thread. Thus, the
// amount of work done in a subscriber's callback should be very minimal.
//
// TODO(b/25725560): Consider having a thread pool for invoking callbacks.
//
// This implementation is single-binary and does not communicate across tasks.

#ifndef TENSORFLOW_SERVING_UTIL_EVENT_BUS_H_
#define TENSORFLOW_SERVING_UTIL_EVENT_BUS_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// Note that the types used for typename E in EventBus must be moveable and
// thread-safe. Future implementations may read Events of type E in multiple
// threads.
template <typename E>
class EventBus : public std::enable_shared_from_this<EventBus<E>> {
  static_assert(std::is_move_assignable<E>::value, "E must be moveable");

 public:
  // Subscription is an RAII object and tracks the lifecycle of a single
  // subscription to an EventBus. Upon destruction, it automatically
  // Unsubscribes from the originating EventBus.
  //
  // Note that Subscription only maintains weak_ptr references to the EventBus,
  // such that the EventBus and Subscriptions can be safely destructed in any
  // order. Subscription is not an owner of EventBus.
  class Subscription {
   public:
    // Unsubscribes the subscriber.
    ~Subscription();

   private:
    friend class EventBus;

    explicit Subscription(std::weak_ptr<EventBus<E>> bus);

    // Weak pointer to the EventBus that originated this Subscription.
    std::weak_ptr<EventBus<E>> bus_;

    TF_DISALLOW_COPY_AND_ASSIGN(Subscription);
  };

  struct Options {
    // The environment to use for time.
    Env* env = Env::Default();
  };

  // Creates an EventBus and returns a shared_ptr to it. This is the only
  // allowed public mechanism for creating an EventBus so that we can track
  // references to an EventBus uniformly.
  static std::shared_ptr<EventBus> CreateEventBus(const Options& options = {});

  ~EventBus() = default;

  // Event and the publish time associated with it.
  struct EventAndTime {
    const E& event;
    uint64 event_time_micros;
  };

  // The function type for EventBus Callbacks to be implemented by clients.
  // Important Warnings:
  // * Callbacks must not themselves callback to the EventBus for any purpose
  //   including subscribing, publishing or unsubscribing. This will cause a
  //   circular deadlock.
  // * Callbacks must do very little work as they are invoked on the
  //   publisher's thread. Any costly work should be performed asynchronously.
  using Callback = std::function<void(const EventAndTime&)>;

  // Subscribes to all events on the EventBus.
  //
  // Returns a Subscription RAII object that can be used to unsubscribe, or will
  // automatically unsubscribe on destruction. Returns a unique_ptr so that we
  // can use the subscription's address to Unsubscribe.
  //
  // Important contract for unsubscribing (deleting the RAII object):
  //   * Unsubscribing (deleting the RAII object) may block while currently
  //     scheduled callback invocation(s) finish.
  //   * Once it returns no callback invocations will occur.
  // Callers' destructors must use the sequence:
  //   (1) Unsubscribe.
  //   (2) Tear down anything that the callback references.
  std::unique_ptr<Subscription> Subscribe(const Callback& callback)
      LOCKS_EXCLUDED(mutex_) TF_MUST_USE_RESULT;

  // Publishes an event to all subscribers.
  void Publish(const E& event) LOCKS_EXCLUDED(mutex_);

 private:
  explicit EventBus(const Options& options);

  // Unsubscribes the specified subscriber. Called only by Subscription.
  void Unsubscribe(const Subscription* subscription) LOCKS_EXCLUDED(mutex_);

  // All of the information needed for a single subscription, both for
  // publishing events and unsubscribing.
  struct SubscriptionTuple {
    // Uniquely identifies the Subscription.
    Subscription* subscription;
    Callback callback;
  };

  // Mutex held for all operations on an EventBus including all publishing and
  // subscription operations.
  mutable mutex mutex_;

  // All subscriptions that the EventBus is aware of. Note that this is not
  // optimized for high scale in the number of subscribers.
  std::vector<SubscriptionTuple> subscriptions_ GUARDED_BY(mutex_);

  const Options options_;

  TF_DISALLOW_COPY_AND_ASSIGN(EventBus);
};

// --- Implementation details below ---

template <typename E>
EventBus<E>::Subscription::Subscription(std::weak_ptr<EventBus<E>> bus)
    : bus_(std::move(bus)) {}

template <typename E>
EventBus<E>::Subscription::~Subscription() {
  std::shared_ptr<EventBus<E>> temp_shared_ptr = bus_.lock();
  if (temp_shared_ptr != nullptr) {
    temp_shared_ptr->Unsubscribe(this);
  }
}

template <typename E>
std::unique_ptr<typename EventBus<E>::Subscription> EventBus<E>::Subscribe(
    const Callback& callback) {
  mutex_lock lock(mutex_);
  std::unique_ptr<Subscription> subscription(
      new Subscription(this->shared_from_this()));
  subscriptions_.push_back({subscription.get(), callback});
  return subscription;
}

template <typename E>
EventBus<E>::EventBus(const Options& options) : options_(options) {}

template <typename E>
std::shared_ptr<EventBus<E>> EventBus<E>::CreateEventBus(
    const Options& options) {
  return std::shared_ptr<EventBus<E>>(new EventBus<E>(options));
}

template <typename E>
void EventBus<E>::Unsubscribe(
    const typename EventBus<E>::Subscription* subscription) {
  mutex_lock lock(mutex_);
  subscriptions_.erase(
      std::remove_if(subscriptions_.begin(), subscriptions_.end(),
                     [subscription](SubscriptionTuple s) {
                       return s.subscription == subscription;
                     }),
      subscriptions_.end());
}

template <typename E>
void EventBus<E>::Publish(const E& event) {
  mutex_lock lock(mutex_);
  const uint64 event_time = options_.env->NowMicros();
  const EventAndTime event_and_time = {event, event_time};
  for (const SubscriptionTuple& subscription : subscriptions_) {
    subscription.callback(event_and_time);
  }
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_EVENT_BUS_H_
