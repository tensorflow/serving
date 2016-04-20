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

#include "tensorflow_serving/core/aspired_versions_manager.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/source.h"

namespace tensorflow {
namespace serving {

namespace {

// The aspired state stored with every managed servable.
//
// We use a struct here instead of a naked bool because this buys some type
// safety, and cannot be implicitly cast to/from pointers by mistake.
struct Aspired {
  bool is_aspired;
};

// Decides which action amongst the 2 to take. We prefer an unload action over a
// load action.
//
// Note that this returns a strict weak ordering.
struct CompareActions {
 public:
  bool operator()(const optional<AspiredVersionPolicy::ServableAction>& lhs,
                  const optional<AspiredVersionPolicy::ServableAction>& rhs) {
    if (!lhs) {
      return false;
    }
    if (!rhs) {
      return true;
    }
    // By this point, we are sure the optionals have values.
    return OrderActions(lhs.value(), rhs.value()).action != rhs.value().action;
  }

 private:
  AspiredVersionPolicy::ServableAction OrderActions(
      const AspiredVersionPolicy::ServableAction& lhs,
      const AspiredVersionPolicy::ServableAction& rhs) {
    switch (lhs.action) {
      case AspiredVersionPolicy::Action::kUnload:
        return lhs;
      case AspiredVersionPolicy::Action::kLoad:
        if (rhs.action == AspiredVersionPolicy::Action::kUnload) {
          return rhs;
        }
        return lhs;
    }
  }
};

}  // namespace

namespace internal {

// AspiredVersionManager's implementation of the Target API.
class AspiredVersionsManagerTargetImpl
    : public TargetBase<std::unique_ptr<Loader>> {
 public:
  explicit AspiredVersionsManagerTargetImpl(
      AspiredVersionsManager* const parent)
      : parent_(parent) {}
  ~AspiredVersionsManagerTargetImpl() override = default;

 protected:
  void SetAspiredVersions(
      const StringPiece servable_name,
      std::vector<ServableData<std::unique_ptr<Loader>>> versions) override {
    parent_->SetAspiredVersions(servable_name, std::move(versions));
  }

 private:
  // A pointer to the manager whose Target implementation this is.
  AspiredVersionsManager* const parent_;

  TF_DISALLOW_COPY_AND_ASSIGN(AspiredVersionsManagerTargetImpl);
};

}  // namespace internal

Status AspiredVersionsManager::Create(
    Options options, std::unique_ptr<AspiredVersionsManager>* manager) {
  if (options.aspired_version_policy == nullptr) {
    return errors::InvalidArgument(
        "AspiredVersionsManager::Options aspired_version_policy must be "
        "non-null");
  }
  BasicManager::Options basic_manager_options;
  basic_manager_options.resource_tracker = std::move(options.resource_tracker);
  basic_manager_options.num_load_unload_threads =
      options.num_load_unload_threads;
  basic_manager_options.max_num_load_retries = options.max_num_load_retries;
  basic_manager_options.load_retry_interval_micros =
      options.load_retry_interval_micros;
  basic_manager_options.servable_event_bus = options.servable_event_bus;
  std::unique_ptr<BasicManager> basic_manager;
  TF_RETURN_IF_ERROR(
      BasicManager::Create(std::move(basic_manager_options), &basic_manager));

  manager->reset(new AspiredVersionsManager(
      options.manage_state_interval_micros, options.env,
      std::move(options.aspired_version_policy), std::move(basic_manager)));
  return Status::OK();
}

AspiredVersionsManager::AspiredVersionsManager(
    int64 manage_state_interval_micros, Env* env,
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy,
    std::unique_ptr<BasicManager> basic_manager)
    : aspired_version_policy_(std::move(aspired_version_policy)),
      target_impl_(new internal::AspiredVersionsManagerTargetImpl(this)),
      basic_manager_(std::move(basic_manager)) {
  if (manage_state_interval_micros > 0) {
    PeriodicFunction::Options pf_options;
    pf_options.env = env;
    pf_options.thread_name_prefix = "AspiredVersionsManager_ManageState_Thread";
    manage_state_thread_.reset(new PeriodicFunction(
        [this]() { this->ManageState(); }, manage_state_interval_micros));
  }
}

AspiredVersionsManager::~AspiredVersionsManager() {
  // This will wait till the thread is joined.
  manage_state_thread_.reset();
}

std::vector<ServableId> AspiredVersionsManager::ListAvailableServableIds()
    const {
  return basic_manager_->ListAvailableServableIds();
}

Status AspiredVersionsManager::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  return basic_manager_->GetUntypedServableHandle(request, untyped_handle);
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
AspiredVersionsManager::GetAvailableUntypedServableHandles() const {
  return basic_manager_->GetAvailableUntypedServableHandles();
}

Source<std::unique_ptr<Loader>>::AspiredVersionsCallback
AspiredVersionsManager::GetAspiredVersionsCallback() {
  return target_impl_->GetAspiredVersionsCallback();
}

void AspiredVersionsManager::SetAspiredVersions(
    const StringPiece servable_name,
    std::vector<ServableData<std::unique_ptr<Loader>>> versions) {
  // We go through the aspired_servable_versions and fill a vector with the
  // next aspired version numbers, and sort it.
  std::vector<int64> next_aspired_versions;
  next_aspired_versions.reserve(versions.size());
  for (const auto& version : versions) {
    if (servable_name != version.id().name) {
      LOG(ERROR) << "Servable name: " << servable_name
                 << " doesn't match name in servable version: "
                 << version.id().name;
      DCHECK(false) << "See previous servable name mismatch error message.";
      return;
    }
    next_aspired_versions.push_back(version.id().version);
  }
  std::sort(next_aspired_versions.begin(), next_aspired_versions.end());

  {
    mutex_lock l(basic_manager_read_modify_write_mu_);

    // We gather all the servables with the servable_name and
    // 1. Add the current aspired version numbers to a vector and sort it,
    // 2. Set the aspired bool to false for all current servable harnesses which
    // are not aspired.
    std::vector<int64> current_aspired_versions;
    for (const ServableStateSnapshot<Aspired> state_snapshot :
         basic_manager_->GetManagedServableStateSnapshots<Aspired>(
             servable_name.ToString())) {
      if (state_snapshot.additional_state->is_aspired) {
        current_aspired_versions.push_back(state_snapshot.id.version);
      }
      // If this version is not part of the aspired versions.
      if (std::find(next_aspired_versions.begin(), next_aspired_versions.end(),
                    state_snapshot.id.version) == next_aspired_versions.end()) {
        basic_manager_->GetAdditionalServableState<Aspired>(state_snapshot.id)
            ->is_aspired = false;
        basic_manager_->CancelLoadServableRetry(state_snapshot.id);
      }
    }
    std::sort(current_aspired_versions.begin(), current_aspired_versions.end());

    // We do a set_difference (A - B), on the next aspired versions and the
    // current aspired versions to find the version numbers which need to be
    // added the harness map.
    std::vector<int64> additions;
    additions.reserve(next_aspired_versions.size());
    std::set_difference(
        next_aspired_versions.begin(), next_aspired_versions.end(),
        current_aspired_versions.begin(), current_aspired_versions.end(),
        std::inserter(additions, additions.begin()));

    // We go through the aspired_servable_versions, pull out the versions which
    // need to be added and add them to the harness map.
    for (auto& version : versions) {
      // if this aspired version is not already present in the map.
      if (std::find(additions.begin(), additions.end(), version.id().version) !=
          additions.end()) {
        basic_manager_->ManageServableWithAdditionalState(
            std::move(version), std::unique_ptr<Aspired>(new Aspired{true}));
      }
    }
  }
}

// We collect the version policy actions for each servable stream first. Then
// we sort them based on the global policy and pick the first one.
optional<AspiredVersionPolicy::ServableAction>
AspiredVersionsManager::GetNextAction() {
  std::vector<optional<AspiredVersionPolicy::ServableAction>> actions;
  std::vector<AspiredServableStateSnapshot> aspired_state_snapshots;
  for (const string& servable_name :
       basic_manager_->GetManagedServableNames()) {
    aspired_state_snapshots.clear();
    for (const ServableStateSnapshot<Aspired>& state_snapshot :
         basic_manager_->GetManagedServableStateSnapshots<Aspired>(
             servable_name)) {
      aspired_state_snapshots.push_back(
          {state_snapshot.id, state_snapshot.state,
           state_snapshot.additional_state->is_aspired});
      actions.emplace_back(
          aspired_version_policy_->GetNextAction(aspired_state_snapshots));
    }
  }

  std::sort(actions.begin(), actions.end(), CompareActions());
  const optional<AspiredVersionPolicy::ServableAction> next_action =
      !actions.empty() ? actions[0] : nullopt;
  return next_action;
}

void AspiredVersionsManager::PerformAction(
    const AspiredVersionPolicy::ServableAction action) {
  switch (action.action) {
    case AspiredVersionPolicy::Action::kLoad: {
      basic_manager_->LoadServable(action.id, [action](const Status& status) {
        if (!status.ok()) {
          LOG(ERROR) << "Servable " << action.id.DebugString()
                     << " cannot be loaded: " << status;
        }
      });
    } break;
    case AspiredVersionPolicy::Action::kUnload: {
      basic_manager_->UnloadServable(action.id, [action](const Status& status) {
        if (!status.ok()) {
          LOG(ERROR) << "Servable " << action.id.DebugString()
                     << " cannot be unloaded: " << status;
        }
      });
    } break;
  }
}

void AspiredVersionsManager::ManageState() {
  mutex_lock l(basic_manager_read_modify_write_mu_);

  const optional<AspiredVersionPolicy::ServableAction> next_action =
      GetNextAction();
  if (!next_action) {
    return;
  }
  // NOTE: we could do action validation here.

  PerformAction(*next_action);
}

}  // namespace serving
}  // namespace tensorflow
