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

#include "tensorflow_serving/core/load_servables_fast.h"

#include <map>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/target.h"

namespace tensorflow {
namespace serving {

namespace internal {

Status ConnectSourceWithFastInitialLoad(
    AspiredVersionsManager* manager, Source<std::unique_ptr<Loader>>* source,
    const std::function<Status()>& wait_until_loaded_fn,
    const uint32 num_threads) {
  const uint32 prev_num_load_threads = manager->num_load_threads();
  manager->SetNumLoadThreads(num_threads);
  ConnectSourceToTarget(source, manager);
  const Status status = wait_until_loaded_fn();
  manager->SetNumLoadThreads(prev_num_load_threads);
  return status;
}

}  // namespace internal

Status ConnectSourceWithFastInitialLoad(
    AspiredVersionsManager* manager, Source<std::unique_ptr<Loader>>* source,
    ServableStateMonitor* servable_state_monitor,
    const std::vector<ServableRequest>& servables, const uint32 num_threads) {
  return internal::ConnectSourceWithFastInitialLoad(
      manager, source,
      [&]() {
        std::map<ServableId, ServableState::ManagerState> states_reached;
        const bool all_servables_available =
            servable_state_monitor->WaitUntilServablesReachState(
                servables, ServableState::ManagerState::kAvailable,
                &states_reached);
        if (!all_servables_available) {
          string message = "Some models did not become available: {";
          for (const auto& id_and_state : states_reached) {
            if (id_and_state.second !=
                ServableState::ManagerState::kAvailable) {
              strings::StrAppend(&message, id_and_state.first.DebugString(),
                                 ", ");
            }
          }
          strings::StrAppend(&message, "}");
          return errors::Unknown(message);
        }
        return Status::OK();
      },
      num_threads);
}

}  // namespace serving
}  // namespace tensorflow
