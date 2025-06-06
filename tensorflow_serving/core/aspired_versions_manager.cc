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
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/context.h"
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
  explicit CompareActions(
      AspiredVersionsManager::CustomSortActionsFn custom_sort_actions)
      : custom_sort_actions_(custom_sort_actions) {}

  bool operator()(
      const absl::optional<AspiredVersionPolicy::ServableAction>& lhs,
      const absl::optional<AspiredVersionPolicy::ServableAction>& rhs) {
    if (!lhs) {
      return false;
    }
    if (!rhs) {
      return true;
    }
    // By this point, we are sure the optionals have values.
    if (custom_sort_actions_) {
      return custom_sort_actions_(lhs.value(), rhs.value());
    }
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
  AspiredVersionsManager::CustomSortActionsFn custom_sort_actions_;
};

// Validates whether all entries in 'versions' pertain to the servable named
// 'servable_name'.
absl::Status ValidateAspiredVersions(
    const absl::string_view servable_name,
    const std::vector<ServableData<std::unique_ptr<Loader>>>& versions) {
  for (const auto& version : versions) {
    if (servable_name != version.id().name) {
      return errors::InvalidArgument(strings::StrCat(
          "Servable name: ", servable_name,
          " doesn't match name in servable version: ", version.id().name));
    }
  }
  return absl::OkStatus();
}

// Returns the set of version numbers in 'versions'.
std::set<int64_t> GetVersionNumbers(
    const std::vector<ServableData<std::unique_ptr<Loader>>>& versions) {
  std::set<int64_t> version_numbers;
  for (const auto& version : versions) {
    version_numbers.insert(version.id().version);
  }
  return version_numbers;
}

// Creates a debug string for a given vector of servable versions.
string ServableVersionsDebugString(
    const std::vector<ServableData<std::unique_ptr<Loader>>>& versions) {
  std::vector<string> version_strings;
  version_strings.reserve(versions.size());
  for (const ServableData<std::unique_ptr<Loader>>& version : versions) {
    version_strings.push_back(version.id().DebugString());
  }
  return absl::StrJoin(version_strings, ", ");
}

}  // namespace

namespace internal {

// AspiredVersionsManager's implementation of the Target API.
class AspiredVersionsManagerTargetImpl final
    : public TargetBase<std::unique_ptr<Loader>> {
 public:
  explicit AspiredVersionsManagerTargetImpl(
      AspiredVersionsManager* const parent)
      : parent_(parent) {}
  ~AspiredVersionsManagerTargetImpl() override { Detach(); }

 protected:
  void SetAspiredVersions(
      const absl::string_view servable_name,
      std::vector<ServableData<std::unique_ptr<Loader>>> versions) override {
    parent_->EnqueueAspiredVersionsRequest(servable_name, std::move(versions));
  }

 private:
  // A pointer to the manager whose Target implementation this is.
  AspiredVersionsManager* const parent_;

  TF_DISALLOW_COPY_AND_ASSIGN(AspiredVersionsManagerTargetImpl);
};

}  // namespace internal

absl::Status AspiredVersionsManager::Create(
    Options options, std::unique_ptr<AspiredVersionsManager>* manager) {
  if (options.aspired_version_policy == nullptr) {
    return errors::InvalidArgument(
        "AspiredVersionsManager::Options aspired_version_policy must be "
        "non-null");
  }
  BasicManager::Options basic_manager_options;
  basic_manager_options.resource_tracker = std::move(options.resource_tracker);
  basic_manager_options.num_load_threads = options.num_load_threads;
  basic_manager_options.num_unload_threads = options.num_unload_threads;
  basic_manager_options.max_num_load_retries = options.max_num_load_retries;
  basic_manager_options.load_retry_interval_micros =
      options.load_retry_interval_micros;
  basic_manager_options.flush_filesystem_caches =
      options.flush_filesystem_caches;
  basic_manager_options.env = options.env;
  basic_manager_options.servable_event_bus = options.servable_event_bus;
  basic_manager_options.pre_load_hook = std::move(options.pre_load_hook);
  if (options.should_retry_model_load) {
    basic_manager_options.should_retry_model_load =
        std::move(options.should_retry_model_load);
  }
  std::unique_ptr<BasicManager> basic_manager;
  TF_RETURN_IF_ERROR(
      BasicManager::Create(std::move(basic_manager_options), &basic_manager));

  manager->reset(new AspiredVersionsManager(
      options.manage_state_interval_micros, options.env,
      std::move(options.aspired_version_policy),
      std::move(options.custom_sort_actions), std::move(basic_manager),
      options.with_current_context));
  (manager->get())->enable_reload_servables_with_error_ =
      options.enable_reload_servables_with_error;
  return absl::OkStatus();
}

AspiredVersionsManager::AspiredVersionsManager(
    int64_t manage_state_interval_micros, Env* env,
    std::unique_ptr<AspiredVersionPolicy> aspired_version_policy,
    CustomSortActionsFn custom_sort_actions,
    std::unique_ptr<BasicManager> basic_manager, bool with_current_context)
    : aspired_version_policy_(std::move(aspired_version_policy)),
      custom_sort_actions_(std::move(custom_sort_actions)),
      target_impl_(new internal::AspiredVersionsManagerTargetImpl(this)),
      basic_manager_(std::move(basic_manager)) {
  set_num_load_threads_observer_.reset(
      new Observer<const uint32>([this](const uint32 num_load_threads) {
        this->SetNumLoadThreads(num_load_threads);
      }));
  if (manage_state_interval_micros > 0) {
    PeriodicFunction::Options pf_options;
    pf_options.env = env;
    pf_options.thread_name_prefix = "AspiredVersionsManager_ManageState_Thread";
    if (with_current_context) {
      tensorflow::Context context(tensorflow::ContextKind::kThread);
      manage_state_thread_.reset(new PeriodicFunction(
          [this, context = std::move(context)]() {
            tensorflow::WithContext wc(context);
            this->FlushServables();
            this->HandlePendingAspiredVersionsRequests();
            this->InvokePolicyAndExecuteAction();
          },
          manage_state_interval_micros));
    } else {
      manage_state_thread_.reset(new PeriodicFunction(
          [this]() {
            this->FlushServables();
            this->HandlePendingAspiredVersionsRequests();
            this->InvokePolicyAndExecuteAction();
          },
          manage_state_interval_micros));
    }
  }
}

AspiredVersionsManager::~AspiredVersionsManager() {
  // Shut off incoming aspired-versions calls. It is important to do this before
  // tearing down any other manager state.
  target_impl_.reset();

  // This will wait till the thread is joined.
  manage_state_thread_.reset();
}

std::vector<ServableId> AspiredVersionsManager::ListAvailableServableIds()
    const {
  return basic_manager_->ListAvailableServableIds();
}

absl::Status AspiredVersionsManager::GetUntypedServableHandle(
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

void AspiredVersionsManager::EnqueueAspiredVersionsRequest(
    const absl::string_view servable_name,
    std::vector<ServableData<std::unique_ptr<Loader>>> versions) {
  const absl::Status validation_status =
      ValidateAspiredVersions(servable_name, versions);
  DCHECK(validation_status.ok()) << validation_status.message();
  if (!validation_status.ok()) {
    LOG(ERROR) << validation_status.message();
    return;
  }

  {
    mutex_lock l(pending_aspired_versions_requests_mu_);
    VLOG(2) << "Enqueueing aspired versions request: " << servable_name << ": "
            << ServableVersionsDebugString(versions);
    pending_aspired_versions_requests_[string(servable_name)] =
        std::move(versions);
  }
}

void AspiredVersionsManager::ProcessAspiredVersionsRequest(
    const absl::string_view servable_name,
    std::vector<ServableData<std::unique_ptr<Loader>>> versions) {
  VLOG(2) << "Processing aspired versions request: " << servable_name << ": "
          << ServableVersionsDebugString(versions);

  const std::set<int64_t> next_aspired_versions = GetVersionNumbers(versions);

  // We gather all the servables with the servable_name and
  // 1. Add the current aspired version numbers to a set,
  // 2. Set the aspired bool to false for all current servable harnesses which
  // are not aspired.
  std::set<int64_t> current_aspired_versions;
  std::set<int64_t> current_aspired_versions_with_error;
  const std::vector<ServableStateSnapshot<Aspired>> state_snapshots =
      basic_manager_->GetManagedServableStateSnapshots<Aspired>(
          string(servable_name));
  for (const ServableStateSnapshot<Aspired>& state_snapshot : state_snapshots) {
    if (state_snapshot.additional_state->is_aspired) {
      current_aspired_versions.insert(state_snapshot.id.version);
      if (state_snapshot.state == LoaderHarness::State::kError) {
        current_aspired_versions_with_error.insert(state_snapshot.id.version);
      }
    }
    // If this version is not part of the aspired versions.
    if (next_aspired_versions.find(state_snapshot.id.version) ==
        next_aspired_versions.end()) {
      VLOG(1) << "Setting is_aspired=false for " << state_snapshot.id;
      basic_manager_->GetAdditionalServableState<Aspired>(state_snapshot.id)
          ->is_aspired = false;
      basic_manager_->CancelLoadServableRetry(state_snapshot.id);
    }
  }

  // We do a set_difference (A - B), on the next aspired versions and the
  // current aspired versions to find the version numbers which need to be
  // added the harness map.
  std::set<int64_t> additions;
  std::set_difference(
      next_aspired_versions.begin(), next_aspired_versions.end(),
      current_aspired_versions.begin(), current_aspired_versions.end(),
      std::inserter(additions, additions.begin()));

  // We go through the aspired_servable_versions, pull out the versions which
  // need to be added and add them to the harness map.
  for (auto& version : versions) {
    bool should_add = false;
    const auto& version_id = version.id();
    if (additions.find(version.id().version) != additions.end()) {
      should_add = true;
    }
    if (enable_reload_servables_with_error_ &&
        current_aspired_versions_with_error.find(version.id().version) !=
            current_aspired_versions_with_error.end()) {
      ServableId id;
      id.name = std::string(servable_name);
      id.version = version_id.version;
      const absl::Status manage_status =
          basic_manager_->StopManagingServable(id);
      DCHECK(manage_status.ok()) << manage_status.message();
      if (!manage_status.ok()) {
        LOG(ERROR) << "Internal error: Unable to clear errored servable "
                   << version_id.DebugString()
                   << " from 'basic_manager_': " << manage_status.message();
      }
      should_add = true;
    }

    // if this aspired version is not already present in the map.
    if (should_add) {
      const absl::Status manage_status =
          basic_manager_->ManageServableWithAdditionalState(
              std::move(version), std::unique_ptr<Aspired>(new Aspired{true}));
      DCHECK(manage_status.ok()) << manage_status.message();
      if (!manage_status.ok()) {
        LOG(ERROR) << "Internal error: Unable to transfer servable "
                   << version_id.DebugString()
                   << " to 'basic_manager_': " << manage_status.message();
      }
    }
  }
}

bool AspiredVersionsManager::ContainsAnyReaspiredVersions(
    const absl::string_view servable_name,
    const std::vector<ServableData<std::unique_ptr<Loader>>>& versions) const {
  const std::vector<ServableStateSnapshot<Aspired>> state_snapshots =
      basic_manager_->GetManagedServableStateSnapshots<Aspired>(
          string(servable_name));
  const std::set<int64_t> version_numbers = GetVersionNumbers(versions);
  for (const ServableStateSnapshot<Aspired>& state_snapshot : state_snapshots) {
    if (!state_snapshot.additional_state->is_aspired &&
        version_numbers.find(state_snapshot.id.version) !=
            version_numbers.end()) {
      return true;
    }
  }
  return false;
}

// We collect the version policy actions for each servable stream first. Then
// we sort them based on the global policy and pick the first one.
absl::optional<AspiredVersionPolicy::ServableAction>
AspiredVersionsManager::GetNextAction() {
  std::vector<absl::optional<AspiredVersionPolicy::ServableAction>> actions;
  for (const string& servable_name :
       basic_manager_->GetManagedServableNames()) {
    std::vector<AspiredServableStateSnapshot> aspired_state_snapshots;
    for (const ServableStateSnapshot<Aspired>& state_snapshot :
         basic_manager_->GetManagedServableStateSnapshots<Aspired>(
             servable_name)) {
      aspired_state_snapshots.push_back(
          {state_snapshot.id, state_snapshot.state,
           state_snapshot.additional_state->is_aspired});
    }
    actions.emplace_back(
        aspired_version_policy_->GetNextAction(aspired_state_snapshots));
  }

  std::sort(actions.begin(), actions.end(),
            CompareActions(custom_sort_actions_));
  const absl::optional<AspiredVersionPolicy::ServableAction> next_action =
      !actions.empty() ? actions[0] : absl::nullopt;
  if (next_action) {
    VLOG(1) << "Taking action: " << next_action->DebugString();
  }
  return next_action;
}

void AspiredVersionsManager::PerformAction(
    const AspiredVersionPolicy::ServableAction action) {
  switch (action.action) {
    case AspiredVersionPolicy::Action::kLoad: {
      basic_manager_->LoadServable(
          action.id, [action](const absl::Status& status) {
            if (!status.ok()) {
              LOG(ERROR) << "Servable " << action.id.DebugString()
                         << " cannot be loaded: " << status;
            }
          });
    } break;
    case AspiredVersionPolicy::Action::kUnload: {
      basic_manager_->UnloadServable(
          action.id, [action](const absl::Status& status) {
            if (!status.ok()) {
              LOG(ERROR) << "Servable " << action.id.DebugString()
                         << " cannot be unloaded: " << status;
            }
          });
    } break;
  }
}

void AspiredVersionsManager::FlushServables() {
  mutex_lock l(basic_manager_read_modify_write_mu_);
  for (const string& servable_name :
       basic_manager_->GetManagedServableNames()) {
    for (const ServableStateSnapshot<Aspired>& state_snapshot :
         basic_manager_->GetManagedServableStateSnapshots<Aspired>(
             servable_name)) {
      if ((state_snapshot.state == LoaderHarness::State::kNew ||
           state_snapshot.state == LoaderHarness::State::kDisabled ||
           state_snapshot.state == LoaderHarness::State::kError) &&
          !state_snapshot.additional_state->is_aspired) {
        const absl::Status status =
            basic_manager_->StopManagingServable(state_snapshot.id);
        if (status.ok()) {
          VLOG(1) << "Removed " << state_snapshot.id << "from BasicManager";
        } else {
          // This scenario is likely a bug, perhaps a race (either here in
          // AspiredVersionsManager, or in BasicManager). We'll wind up retrying
          // StopManagingServable() on the next FlushServables() call, so just
          // log the error and move on for now.
          LOG(ERROR) << "Error removing " << state_snapshot.id
                     << "from BasicManager: " << status << " will retry later";
        }
      }
    }
  }
}

void AspiredVersionsManager::HandlePendingAspiredVersionsRequests() {
  mutex_lock l(basic_manager_read_modify_write_mu_);
  mutex_lock l2(pending_aspired_versions_requests_mu_);

  // To be able to process an aspired-versions request, we wait for any
  // re-aspired versions (versions not currently marked aspired, but present in
  // the latest aspired-versions request) to quiesce and be removed from
  // BasicManager. If an enqueued request does contain re-aspired versions, we
  // simply leave it in the queue for now.
  for (auto it = pending_aspired_versions_requests_.begin();
       it != pending_aspired_versions_requests_.end();) {
    const string& servable_name = it->first;
    std::vector<ServableData<std::unique_ptr<Loader>>>& versions = it->second;

    if (ContainsAnyReaspiredVersions(servable_name, versions)) {
      // Sit on it for now. We'll check again later.
      ++it;
      VLOG(1) << "Postponing processing of aspired versions request due to "
                 "re-aspired version(s) among: "
              << ServableVersionsDebugString(versions);
    } else {
      ProcessAspiredVersionsRequest(servable_name, std::move(versions));
      it = pending_aspired_versions_requests_.erase(it);
    }
  }
}

void AspiredVersionsManager::InvokePolicyAndExecuteAction() {
  mutex_lock l(basic_manager_read_modify_write_mu_);

  const absl::optional<AspiredVersionPolicy::ServableAction> next_action =
      GetNextAction();
  if (!next_action) {
    return;
  }
  // NOTE: we could do action validation here.

  PerformAction(*next_action);
}

void AspiredVersionsManager::SetNumLoadThreads(const uint32 num_load_threads) {
  basic_manager_->SetNumLoadThreads(num_load_threads);
}

uint32 AspiredVersionsManager::num_load_threads() const {
  return basic_manager_->num_load_threads();
}

}  // namespace serving
}  // namespace tensorflow
