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

// Builds sets of specific and latest servable requests.
void ExtractRequestSets(const std::vector<ServableRequest>& requests,
                        std::set<ServableId>* specific_servable_requests,
                        std::set<string>* latest_servable_requests) {
  for (const ServableRequest& request : requests) {
    if (request.version) {
      specific_servable_requests->emplace(
          ServableId{request.name, request.version.value()});
    } else {
      latest_servable_requests->emplace(request.name);
    }
  }
}

// Implements the test logic of WaitUntilServablesAvailableForRequests().
bool ServablesAvailableForRequests(
    const std::set<ServableId>& specific_servable_requests,
    const std::set<string>& latest_servable_requests, Manager* const manager) {
  // Build sets of the specific and latest servables available in the manager.
  const std::vector<ServableId> servables_list =
      manager->ListAvailableServableIds();
  const std::set<ServableId> specific_available_servables = {
      servables_list.begin(), servables_list.end()};
  std::set<string> latest_available_servables;
  for (const ServableId& id : specific_available_servables) {
    latest_available_servables.insert(id.name);
  }

  // Check that the specific available servables include the requested
  // servables.
  const bool specific_servables_available = std::includes(
      specific_available_servables.begin(), specific_available_servables.end(),
      specific_servable_requests.begin(), specific_servable_requests.end());

  // Check that the latest available servables includes the requested
  // servables.
  const bool latest_servables_available = std::includes(
      latest_available_servables.begin(), latest_available_servables.end(),
      latest_servable_requests.begin(), latest_servable_requests.end());

  return specific_servables_available && latest_servables_available;
}

}  // namespace

void WaitUntilServablesAvailableForRequests(
    const std::vector<ServableRequest>& requests, Manager* const manager) {
  // The subset of requests that require a specific version.
  std::set<ServableId> specific_servable_requests;
  // The subset of requests that do not require a specific version i.e. require
  // the latest version.
  std::set<string> latest_servable_requests;

  ExtractRequestSets(requests, &specific_servable_requests,
                     &latest_servable_requests);
  while (!ServablesAvailableForRequests(specific_servable_requests,
                                        latest_servable_requests, manager)) {
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
