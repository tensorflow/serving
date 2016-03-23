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

#include "tensorflow_serving/core/availability_helpers.h"

#include <algorithm>
#include <iterator>
#include <set>

#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/core/manager.h"

namespace tensorflow {
namespace serving {
namespace {

// Implements the test logic of WaitUntilServablesAvailableForRequests().
bool ServablesAvailableForRequests(
    const std::vector<ServableRequest>& servables, Manager* const manager) {
  // The subset of 'servables' that require a specific version
  std::set<ServableId> query_specific_servables;
  // The subset of 'servables' that do not require a specific version
  std::set<string> query_latest_servables;

  // Build sets of the query specific servables and latest servables
  for (const ServableRequest& request : servables) {
    if (request.version) {
      query_specific_servables.emplace(
          ServableId{request.name, request.version.value()});
    } else {
      query_latest_servables.emplace(request.name);
    }
  }

  // Build sets of the available specific servables and latest servables
  const std::vector<ServableId> available_servables_list =
      manager->ListAvailableServableIds();
  const std::set<ServableId> available_specific_servables = {
      available_servables_list.begin(), available_servables_list.end()};

  std::set<string> available_latest_servables;
  for (const ServableId& id : available_specific_servables) {
    available_latest_servables.insert(id.name);
  }

  // Check that the available specific servables includes the query's.
  const bool specific_servables_available = std::includes(
      available_specific_servables.begin(), available_specific_servables.end(),
      query_specific_servables.begin(), query_specific_servables.end());

  // Check that the available latest servables includes the query's.
  const bool latest_servables_available = std::includes(
      available_latest_servables.begin(), available_latest_servables.end(),
      query_latest_servables.begin(), query_latest_servables.end());

  return specific_servables_available && latest_servables_available;
}

}  // namespace

void WaitUntilServablesAvailableForRequests(
    const std::vector<ServableRequest>& servables, Manager* const manager) {
  while (!ServablesAvailableForRequests(servables, manager)) {
    Env::Default()->SleepForMicroseconds(50 * 1000 /* 50 ms */);
  }
}

void WaitUntilServablesAvailable(const std::set<ServableId>& servables,
                                 Manager* const manager) {
  std::vector<ServableRequest> requests;
  for (const ServableId& id : servables) {
    requests.push_back(ServableRequest::FromId(id));
  }
  return WaitUntilServablesAvailableForRequests(requests, manager);
}

}  // namespace serving
}  // namespace tensorflow
