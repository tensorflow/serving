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
    DCHECK(!state_and_time.state.health.ok())
        << "Servable: " << state_and_time
        << " is not in error and directly went to state kEnd.";
    return;
  }
  auto& version_map = servable_map_it->second;
  auto version_map_it = version_map.find(version);
  if (version_map_it == version_map.end()) {
    DCHECK(!state_and_time.state.health.ok())
        << "Servable: " << state_and_time
        << " is not in error and directly went to state kEnd.";
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

optional<ServableStateMonitor::ServableStateAndTime>
ServableStateMonitor::GetStateAndTime(const ServableId& servable_id) const {
  mutex_lock l(mu_);

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
  if (options_.max_count_log_events == 0) {
    return;
  }
  while (log_.size() >= options_.max_count_log_events) {
    log_.pop_front();
  }
  log_.emplace_back(state_and_time.state, state_and_time.event_time_micros);
}

}  // namespace serving
}  // namespace tensorflow
