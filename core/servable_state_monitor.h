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

#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_STATE_MONITOR_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_STATE_MONITOR_H_

#include <deque>
#include <functional>
#include <map>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// A utility that listens to an EventBus<ServableState>, and keeps track of the
// state of each servable mentioned on the bus. The intended use case is to
// track the states of servables in a Manager.
//
// Offers an interface for querying the servable states. It may be useful as the
// basis for dashboards, as well as for testing a manager.
//
// IMPORTANT: You must create this monitor before arranging for events to be
// published on the event bus, e.g. giving the event bus to a Manager.
class ServableStateMonitor {
 public:
  struct ServableStateAndTime {
    ServableStateAndTime() = default;
    ServableStateAndTime(ServableState servable_state, const uint64 event_time)
        : state(std::move(servable_state)), event_time_micros(event_time) {}

    // State of the servable.
    ServableState state;

    // Time at which servable state event was published.
    uint64 event_time_micros;

    // Returns a string representation of this struct useful for debugging or
    // logging.
    string DebugString() const;
  };

  using ServableName = string;
  using Version = int64;
  using VersionMap =
      std::map<Version, ServableStateAndTime, std::greater<Version>>;
  using ServableMap = std::map<ServableName, VersionMap>;

  struct Options {
    Options() {}

    // Upper bound for the number of events captured in the bounded log. If set
    // to 0, logging is disabled.
    uint64 max_count_log_events = 0;
  };
  using BoundedLog = std::deque<ServableStateAndTime>;

  explicit ServableStateMonitor(EventBus<ServableState>* bus,
                                const Options& options = Options());
  virtual ~ServableStateMonitor();

  // Returns the current state of one servable, or nullopt if that servable is
  // not being tracked.
  optional<ServableState> GetState(const ServableId& servable_id) const
      LOCKS_EXCLUDED(mu_);

  // Returns the current state and time of one servable, or nullopt if that
  // servable is not being tracked.
  optional<ServableStateAndTime> GetStateAndTime(
      const ServableId& servable_id) const LOCKS_EXCLUDED(mu_);

  // Returns the current states of all tracked versions of the given servable,
  // if any.
  VersionMap GetVersionStates(const string& servable_name) const
      LOCKS_EXCLUDED(mu_);

  // Returns the current states of all tracked versions of all servables.
  ServableMap GetAllServableStates() const LOCKS_EXCLUDED(mu_);

  // Returns the current states of all versions of all servables which have not
  // transitioned to state ServableState::ManagerState::kEnd.
  ServableMap GetLiveServableStates() const LOCKS_EXCLUDED(mu_);

  // Returns the current bounded log of handled servable state events.
  BoundedLog GetBoundedLog() const LOCKS_EXCLUDED(mu_);

  // Notifies when all of the servables have reached the 'goal_state'.
  //
  // Servables can be specified in two ways:
  //   1. As specific versions of a servable stream name. In this case, we check
  //   whether the specific version has reached the 'goal_state' or kEnd.
  //   2. As latest versions, in which case any version for a servable stream
  //   name will be matched against the 'goal_state' or kEnd.
  //
  // We call the 'notifier_fn' when both conditions are true -
  //   1. All of the specific servable requests have either reached the
  //   'goal_state' or kEnd.
  //   2. All of the latest servable requests have reached 'goal_state' or kEnd.
  // The 'notifier_fn' will be called only once, and not repeatedly.
  //
  // The 'reached_goal_state' argument is set as true iff all of the specific
  // servables have reached 'goal_state'.  So callers should verify that
  // 'reached_goal_state' is true in the 'notifier_fn'.
  //
  // The 'states_reached' argument is populated with the servable's id and the
  // state it reached. The state would be 'goal_state' if 'reached_goal_state'
  // is true, else it will contain one or more servables in kEnd state. For
  // latest servable requests, the servable id will be the id of the servable in
  // the stream which reached the state.
  using ServableStateNotifierFn = std::function<void(
      bool reached_goal_state,
      const std::map<ServableId, ServableState::ManagerState>& states_reached)>;
  void NotifyWhenServablesReachState(
      const std::vector<ServableRequest>& servables,
      ServableState::ManagerState goal_state,
      const ServableStateNotifierFn& notifier_fn) LOCKS_EXCLUDED(mu_);

  // Similar to NotifyWhenServablesReachState(...), but instead of notifying, we
  // wait until the 'goal_state' or kEnd is reached.
  //
  // To understand the return value and the return parameter 'states_reached',
  // please read the documentation on NotifyWhenServablesReachState(...).
  bool WaitUntilServablesReachState(
      const std::vector<ServableRequest>& servables,
      ServableState::ManagerState goal_state,
      std::map<ServableId, ServableState::ManagerState>* states_reached =
          nullptr) LOCKS_EXCLUDED(mu_) TF_MUST_USE_RESULT;

 private:
  optional<ServableStateMonitor::ServableStateAndTime> GetStateAndTimeInternal(
      const ServableId& servable_id) const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Request to send notification, setup using
  // NotifyWhenServablesReachState(...).
  struct ServableStateNotificationRequest {
    std::vector<ServableRequest> servables;
    ServableState::ManagerState goal_state;
    ServableStateNotifierFn notifier_fn;
  };

  // Checks whether the notification request is satisfied and we cand send it.
  // If so, returns the 'reached_goal_state' bool and the 'states_reached' by
  // each servable.  Oterwise returns nullopt.
  optional<std::pair<bool, std::map<ServableId, ServableState::ManagerState>>>
  ShouldSendNotification(
      const ServableStateNotificationRequest& notification_request)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Goes through the notification requests and tries to see if any of them can
  // be sent. If a notification is sent, the corresponding request is removed.
  void MaybeSendNotifications() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // This method is called when an event comes in, but before we update our
  // state with the contents of the event. Subclasses may override this method
  // to do custom prepreocessing based on the event and the previous state of
  // the monitor, like calculate load-time, etc.
  virtual void PreHandleEvent(
      const EventBus<ServableState>::EventAndTime& state_and_time);

  // Handles a bus event.
  void HandleEvent(const EventBus<ServableState>::EventAndTime& state_and_time)
      LOCKS_EXCLUDED(mu_);

  const Options options_;

  std::unique_ptr<EventBus<ServableState>::Subscription> bus_subscription_;

  mutable mutex mu_;

  // The current state of each servable version that has appeared on the bus.
  // (Entries are never removed, even when they enter state kEnd.)
  ServableMap states_ GUARDED_BY(mu_);

  // The current state of each servable version that has not transitioned to
  // state ServableState::ManagerState::kEnd.
  ServableMap live_states_ GUARDED_BY(mu_);

  // Deque of pairs of timestamp and ServableState, corresponding to the most
  // recent servable state events handled by the monitor. The size of this deque
  // is upper bounded by max_count_log_events in Options.
  BoundedLog log_ GUARDED_BY(mu_);

  std::vector<ServableStateNotificationRequest>
      servable_state_notification_requests_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ServableStateMonitor);
};

inline bool operator==(const ServableStateMonitor::ServableStateAndTime& a,
                       const ServableStateMonitor::ServableStateAndTime& b) {
  return a.event_time_micros == b.event_time_micros && a.state == b.state;
}

inline bool operator!=(const ServableStateMonitor::ServableStateAndTime& a,
                       const ServableStateMonitor::ServableStateAndTime& b) {
  return !(a == b);
}

inline std::ostream& operator<<(
    std::ostream& os,
    const ServableStateMonitor::ServableStateAndTime& state_and_time) {
  return os << state_and_time.DebugString();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVABLE_STATE_MONITOR_H_
