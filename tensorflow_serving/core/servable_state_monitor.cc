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

#include "tensorflow_serving/core/servable_state_monitor.h"

#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {
namespace serving {
namespace {

void EraseLiveStatesEntry(
    const ServableStateMonitor::ServableStateAndTime& state_and_time,
    ServableStateMonitor::ServableMap* const live_states) {
  const string& servable_name = state_and_time.state.id.name;
  const int64 version = state_and_time.state.id.version;
  auto servable_map_it = live_states->find(servable_name);
  if (servable_map_it == live_states->end()) {
    return;
  }
  auto& version_map = servable_map_it->second;
  auto version_map_it = version_map.find(version);
  if (version_map_it == version_map.end()) {
    return;
  }

  version_map.erase(version_map_it);
  if (version_map.empty()) {
    live_states->erase(servable_map_it);
  }
}

void UpdateLiveStates(
    const ServableStateMonitor::ServableStateAndTime& state_and_time,
    ServableStateMonitor::ServableMap* const live_states) {
  const string& servable_name = state_and_time.state.id.name;
  const int64 version = state_and_time.state.id.version;
  if (state_and_time.state.manager_state != ServableState::ManagerState::kEnd) {
    (*live_states)[servable_name][version] = state_and_time;
  } else {
    EraseLiveStatesEntry(state_and_time, live_states);
  }
}

// Returns the state reached iff the servable has reached 'goal_state' or kEnd,
// otherwise nullopt.
optional<ServableState::ManagerState> HasSpecificServableReachedState(
    const ServableId& servable_id, const ServableState::ManagerState goal_state,
    const optional<ServableStateMonitor::ServableStateAndTime>
        opt_servable_state_time) {
  if (!opt_servable_state_time) {
    return {};
  }
  const ServableState::ManagerState state =
      opt_servable_state_time->state.manager_state;
  if (state != goal_state && state != ServableState::ManagerState::kEnd) {
    return {};
  }
  return {state};
}

// Returns the id of the servable in the stream which has reached 'goal_state'
// or kEnd. If no servable has done so, returns nullopt.
optional<ServableId> HasAnyServableInStreamReachedState(
    const string& stream_name, const ServableState::ManagerState goal_state,
    const ServableStateMonitor::ServableMap& states) {
  optional<ServableId> opt_servable_id;
  const auto found_it = states.find(stream_name);
  if (found_it == states.end()) {
    return {};
  }
  const ServableStateMonitor::VersionMap& version_map = found_it->second;
  for (const auto& version_and_state_time : version_map) {
    const ServableStateMonitor::ServableStateAndTime& state_and_time =
        version_and_state_time.second;
    if (state_and_time.state.manager_state == goal_state ||
        state_and_time.state.manager_state ==
            ServableState::ManagerState::kEnd) {
      return {version_and_state_time.second.state.id};
    }
  }
  return {};
}

}  // namespace

string ServableStateMonitor::ServableStateAndTime::DebugString() const {
  return strings::StrCat("state: {", state.DebugString(),
                         "}, event_time_micros: ", event_time_micros);
}

ServableStateMonitor::ServableStateMonitor(EventBus<ServableState>* bus,
                                           const Options& options)
    : options_(options),
      bus_subscription_(bus->Subscribe(
          [this](const EventBus<ServableState>::EventAndTime& state_and_time) {
            this->HandleEvent(state_and_time);
          })) {}

ServableStateMonitor::~ServableStateMonitor() {
  // Halt event handling first, before tearing down state that event handling
  // may access such as 'servable_state_notification_requests_'.
  bus_subscription_ = nullptr;
}

optional<ServableStateMonitor::ServableStateAndTime>
ServableStateMonitor::GetStateAndTimeInternal(
    const ServableId& servable_id) const {
  auto it = states_.find(servable_id.name);
  if (it == states_.end()) {
    return nullopt;
  }
  const VersionMap& versions = it->second;
  auto it2 = versions.find(servable_id.version);
  if (it2 == versions.end()) {
    return nullopt;
  }
  return it2->second;
}

optional<ServableStateMonitor::ServableStateAndTime>
ServableStateMonitor::GetStateAndTime(const ServableId& servable_id) const {
  mutex_lock l(mu_);
  return GetStateAndTimeInternal(servable_id);
}

optional<ServableState> ServableStateMonitor::GetState(
    const ServableId& servable_id) const {
  const optional<ServableStateAndTime>& state_and_time =
      GetStateAndTime(servable_id);
  if (!state_and_time) {
    return nullopt;
  }
  return state_and_time->state;
}

ServableStateMonitor::VersionMap ServableStateMonitor::GetVersionStates(
    const string& servable_name) const {
  mutex_lock l(mu_);
  auto it = states_.find(servable_name);
  if (it == states_.end()) {
    return {};
  }
  return it->second;
}

ServableStateMonitor::ServableMap ServableStateMonitor::GetAllServableStates()
    const {
  mutex_lock l(mu_);
  return states_;
}

ServableStateMonitor::ServableMap ServableStateMonitor::GetLiveServableStates()
    const {
  mutex_lock l(mu_);
  return live_states_;
}

ServableStateMonitor::BoundedLog ServableStateMonitor::GetBoundedLog() const {
  mutex_lock l(mu_);
  return log_;
}

void ServableStateMonitor::NotifyWhenServablesReachState(
    const std::vector<ServableRequest>& servables,
    const ServableState::ManagerState goal_state,
    const ServableStateNotifierFn& notifier_fn) {
  mutex_lock l(mu_);
  servable_state_notification_requests_.push_back(
      {servables, goal_state, notifier_fn});
  MaybeSendNotifications();
}

bool ServableStateMonitor::WaitUntilServablesReachState(
    const std::vector<ServableRequest>& servables,
    const ServableState::ManagerState goal_state,
    std::map<ServableId, ServableState::ManagerState>* const states_reached) {
  bool reached_goal_state;
  Notification notified;
  NotifyWhenServablesReachState(
      servables, goal_state,
      [&](const bool incoming_reached_goal_state,
          const std::map<ServableId, ServableState::ManagerState>&
              incoming_states_reached) {
        if (states_reached != nullptr) {
          *states_reached = incoming_states_reached;
        }
        reached_goal_state = incoming_reached_goal_state;
        notified.Notify();
      });
  notified.WaitForNotification();
  return reached_goal_state;
}

void ServableStateMonitor::PreHandleEvent(
    const EventBus<ServableState>::EventAndTime& state_and_time) {}

void ServableStateMonitor::HandleEvent(
    const EventBus<ServableState>::EventAndTime& event_and_time) {
  PreHandleEvent(event_and_time);

  mutex_lock l(mu_);
  const ServableStateAndTime state_and_time = {
      event_and_time.event, event_and_time.event_time_micros};
  states_[state_and_time.state.id.name][state_and_time.state.id.version] =
      state_and_time;
  UpdateLiveStates(state_and_time, &live_states_);
  MaybeSendNotifications();

  if (options_.max_count_log_events == 0) {
    return;
  }
  while (log_.size() >= options_.max_count_log_events) {
    log_.pop_front();
  }
  log_.emplace_back(state_and_time.state, state_and_time.event_time_micros);
}

optional<std::pair<bool, std::map<ServableId, ServableState::ManagerState>>>
ServableStateMonitor::ShouldSendNotification(
    const ServableStateNotificationRequest& notification_request) {
  bool reached_goal_state = true;
  std::map<ServableId, ServableState::ManagerState> states_reached;
  for (const auto& servable_request : notification_request.servables) {
    if (servable_request.version) {
      const ServableId servable_id = {servable_request.name,
                                      *servable_request.version};
      const optional<ServableState::ManagerState> opt_state =
          HasSpecificServableReachedState(servable_id,
                                          notification_request.goal_state,
                                          GetStateAndTimeInternal(servable_id));
      if (!opt_state) {
        return {};
      }
      // Remains false once false.
      reached_goal_state =
          reached_goal_state && *opt_state == notification_request.goal_state;
      states_reached[servable_id] = *opt_state;
    } else {
      const optional<ServableId> opt_servable_id =
          HasAnyServableInStreamReachedState(
              servable_request.name, notification_request.goal_state, states_);
      if (!opt_servable_id) {
        return {};
      }
      const ServableState::ManagerState reached_state =
          GetStateAndTimeInternal(*opt_servable_id)->state.manager_state;
      // Remains false once false.
      reached_goal_state = reached_goal_state &&
                           reached_state == notification_request.goal_state;
      states_reached[*opt_servable_id] = reached_state;
    }
  }
  return {{reached_goal_state, states_reached}};
}

void ServableStateMonitor::MaybeSendNotifications() {
  for (auto iter = servable_state_notification_requests_.begin();
       iter != servable_state_notification_requests_.end();) {
    const ServableStateNotificationRequest& notification_request = *iter;
    const optional<
        std::pair<bool, std::map<ServableId, ServableState::ManagerState>>>
        opt_state_and_states_reached =
            ShouldSendNotification(notification_request);
    if (opt_state_and_states_reached) {
      notification_request.notifier_fn(opt_state_and_states_reached->first,
                                       opt_state_and_states_reached->second);
      iter = servable_state_notification_requests_.erase(iter);
    } else {
      ++iter;
    }
  }
}

}  // namespace serving
}  // namespace tensorflow
