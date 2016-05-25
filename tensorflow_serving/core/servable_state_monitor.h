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
#include <map>

#include "tensorflow/core/platform/env.h"
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
  using VersionMap = std::map<Version, ServableStateAndTime>;
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
  virtual ~ServableStateMonitor() = default;

  // Returns the current state of one servable, or nullopt if that servable is
  // not being tracked.
  optional<ServableState> GetState(const ServableId& servable_id) const;

  // Returns the current state and time of one servable, or nullopt if that
  // servable is not being tracked.
  optional<ServableStateAndTime> GetStateAndTime(
      const ServableId& servable_id) const;

  // Returns the current states of all tracked versions of the given servable,
  // if any.
  VersionMap GetVersionStates(const string& servable_name) const;

  // Returns the current states of all tracked versions of all servables.
  ServableMap GetAllServableStates() const;

  // Returns the current states of all versions of all servables which have not
  // transitioned to state ServableState::ManagerState::kEnd.
  ServableMap GetLiveServableStates() const;

  // Returns the current bounded log of handled servable state events.
  BoundedLog GetBoundedLog() const;

 private:
  // This method is called when an event comes in, but before we update our
  // state with the contents of the event. Subclasses may override this method
  // to do custom prepreocessing based on the event and the previous state of
  // the monitor, like calculate load-time, etc.
  virtual void PreHandleEvent(
      const EventBus<ServableState>::EventAndTime& state_and_time);

  // Handles a bus event.
  void HandleEvent(const EventBus<ServableState>::EventAndTime& state_and_time);

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
