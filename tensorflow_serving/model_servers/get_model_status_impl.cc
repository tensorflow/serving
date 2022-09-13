/* Copyright 2018 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/get_model_status_impl.h"

#include <string>

#include "absl/types/optional.h"
#include "tensorflow_serving/apis/status.pb.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/util/status_util.h"

namespace tensorflow {
namespace serving {

namespace {

// Converts ManagerState to enum State in GetModelStatusResponse.
ModelVersionStatus_State ManagerStateToStateProtoEnum(
    const ServableState::ManagerState& manager_state) {
  switch (manager_state) {
    case ServableState::ManagerState::kStart: {
      return ModelVersionStatus_State_START;
    }
    case ServableState::ManagerState::kLoading: {
      return ModelVersionStatus_State_LOADING;
    }
    case ServableState::ManagerState::kAvailable: {
      return ModelVersionStatus_State_AVAILABLE;
    }
    case ServableState::ManagerState::kUnloading: {
      return ModelVersionStatus_State_UNLOADING;
    }
    case ServableState::ManagerState::kEnd: {
      return ModelVersionStatus_State_END;
    }
  }
}

// Adds ModelVersionStatus to GetModelStatusResponse
void AddModelVersionStatusToResponse(GetModelStatusResponse* response,
                                     const int64& version,
                                     const ServableState& servable_state) {
  ModelVersionStatus* version_status = response->add_model_version_status();
  version_status->set_version(version);
  version_status->set_state(
      ManagerStateToStateProtoEnum(servable_state.manager_state));
  *version_status->mutable_status() = ToStatusProto(servable_state.health);
}

}  // namespace

Status GetModelStatusImpl::GetModelStatus(ServerCore* core,
                                          const GetModelStatusRequest& request,
                                          GetModelStatusResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::errors::InvalidArgument("Missing ModelSpec");
  }
  return GetModelStatusWithModelSpec(core, request.model_spec(), request,
                                     response);
}

Status GetModelStatusImpl::GetModelStatusWithModelSpec(
    ServerCore* core, const ModelSpec& model_spec,
    const GetModelStatusRequest& request, GetModelStatusResponse* response) {
  const string& model_name = model_spec.name();
  const ServableStateMonitor& monitor = *core->servable_state_monitor();

  if (model_spec.has_version()) {
    // Only gets status for specified version of specified model.
    const int64_t version = model_spec.version().value();
    const ServableId id = {model_name, version};
    const absl::optional<ServableState> opt_servable_state =
        monitor.GetState(id);
    if (!opt_servable_state) {
      return tensorflow::errors::NotFound("Could not find version ", version,
                                          " of model ", model_name);
    }
    AddModelVersionStatusToResponse(response, version,
                                    opt_servable_state.value());
  } else {
    // Gets status for all versions of specified model.
    const ServableStateMonitor::VersionMap versions_and_states =
        monitor.GetVersionStates(model_name);
    if (versions_and_states.empty()) {
      return tensorflow::errors::NotFound(
          "Could not find any versions of model ", model_name);
    }
    for (const auto& version_and_state : versions_and_states) {
      const int64_t version = version_and_state.first;
      const ServableState& servable_state = version_and_state.second.state;
      AddModelVersionStatusToResponse(response, version, servable_state);
    }
  }
  return tensorflow::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
