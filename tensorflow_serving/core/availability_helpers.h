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

// Helper methods related to the availability of servables.

#ifndef TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_
#define TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_

#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

// Waits until all of these servables are available for serving through the
// provided manager.
//
// Servables can be specified in two ways:
//   1. As specific versions, in which case one or more specific versions can be
//      specified for a servable stream name, and must match exactly.
//   2. As latest versions, in which case any version available for a servable
//      stream name will be matched.
//
// Implementation notes:
// The method is implemented by polling the manager, and the periodicity of
// polling is 50ms.  We call ListAvailableServableIds on the manager, which may
// have side-effects for certain manager implementations, e.g. causing servables
// to be loaded.
void WaitUntilServablesAvailableForRequests(
    const std::vector<ServableRequest>& servables, Manager* manager);

// Like WaitUntilServablesAvailableForRequests(), but taking a set of servable
// ids (and hence waits for the specific versions specified in the ids).
void WaitUntilServablesAvailable(const std::set<ServableId>& servables,
                                 Manager* manager);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_
