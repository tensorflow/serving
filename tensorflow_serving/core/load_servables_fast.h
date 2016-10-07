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

#ifndef TENSORFLOW_SERVING_CORE_LOAD_SERVABLES_FAST_H_
#define TENSORFLOW_SERVING_CORE_LOAD_SERVABLES_FAST_H_

#include <functional>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_state_monitor.h"

namespace tensorflow {
namespace serving {

// Connects 'source' to 'manager', and speeds up loading of the servables
// matching 'initial_servables'. The speeding up is accomplished by boosting the
// number of threads used for loading until the initial servables have been
// loaded, and then resetting it to the manager's originally configured value.
Status ConnectSourceWithFastInitialLoad(
    AspiredVersionsManager* manager, Source<std::unique_ptr<Loader>>* source,
    ServableStateMonitor* servable_state_monitor,
    const std::vector<ServableRequest>& initial_servables,
    uint32 num_threads = 4 * port::NumSchedulableCPUs());

////
// Implementation detail. API readers may skip.
///

namespace internal {

Status ConnectSourceWithFastInitialLoad(
    AspiredVersionsManager* manager, Source<std::unique_ptr<Loader>>* source,
    const std::function<Status()>& wait_until_loaded_fn, uint32 num_threads);

}  // namespace internal

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_LOAD_SERVABLES_FAST_H_
