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

ServableStateMonitor::ServableStateMonitor(const Options& options,
                                           EventBus<ServableState>* bus)
    : options_(options),
      bus_subscription_(bus->Subscribe(
          [this](const ServableState& state) { this->HandleEvent(state); })) {}

ServableStateMonitor::ServableStateMonitor(EventBus<ServableState>* bus)
    : ServableStateMonitor(Options(), bus) {}

optional<ServableState> ServableStateMonitor::GetState(
    const ServableId& servable_id) const {
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

ServableStateMonitor::BoundedLog ServableStateMonitor::GetBoundedLog() const {
  mutex_lock l(mu_);
  return log_;
}

void ServableStateMonitor::HandleEvent(const ServableState& state) {
  mutex_lock l(mu_);
  states_[state.id.name][state.id.version] = state;
  if (options_.max_count_log_events == 0) {
    return;
  }
  while (log_.size() >= options_.max_count_log_events) {
    log_.pop_front();
  }
  log_.emplace_back(options_.env->NowMicros(), state);
}

}  // namespace serving
}  // namespace tensorflow
