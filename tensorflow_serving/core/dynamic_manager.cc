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

#include "tensorflow_serving/core/dynamic_manager.h"

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

namespace internal {

// DynamicManager's implementation of the Target API.
class DynamicManagerTargetImpl : public TargetBase<std::unique_ptr<Loader>> {
 public:
  explicit DynamicManagerTargetImpl(DynamicManager* parent) : parent_(parent) {}
  ~DynamicManagerTargetImpl() override = default;

 protected:
  void SetAspiredVersions(
      const StringPiece servable_name,
      std::vector<ServableData<std::unique_ptr<Loader>>> versions) override {
    parent_->SetAspiredVersions(servable_name, std::move(versions));
  }

 private:
  // A pointer to the manager whose Target implementation this is.
  DynamicManager* parent_;

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicManagerTargetImpl);
};

}  // namespace internal

namespace {

// Decides which action amongst the 2 to take. We prefer an unload action over a
// load action.
//
// Note that this returns a strict weak ordering.
struct CompareActions {
 public:
  bool operator()(const optional<VersionPolicy::ServableAction>& lhs,
                  const optional<VersionPolicy::ServableAction>& rhs) {
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
  VersionPolicy::ServableAction OrderActions(
      const VersionPolicy::ServableAction& lhs,
      const VersionPolicy::ServableAction& rhs) {
    switch (lhs.action) {
      case VersionPolicy::Action::kUnload:
        return lhs;
      case VersionPolicy::Action::kLoad:
        if (rhs.action == VersionPolicy::Action::kUnload) {
          return rhs;
        }
        return lhs;
    }
  }
};

}  // namespace

struct DynamicManager::ServingMap::EqRequest {
  bool operator()(const ServableRequest& lhs,
                  const ServableRequest& rhs) const {
    if (lhs.version != rhs.version) {
      return false;
    }
    // Even if there is a small probability that version checking can eliminate
    // string checking, we should do that since O(string_equality) >>
    // O(version_equality)
    if (lhs.name != rhs.name) {
      return false;
    }
    return true;
  }
};

struct DynamicManager::ServingMap::HashRequest {
  uint64 operator()(const ServableRequest& request) const {
    // Hash codes for many common types are remarkably bad, often clustering
    // around the same values of the low and/or high bits for linear
    // sequences of inputs such as 1, 2, 3; or addresses of consecutively
    // allocated objects.  For these cases the default hash function is the
    // identity function on the bit patterns.
    //
    // So we apply a one-to-one mapping to the resulting bit patterns to
    // make the high bits contain more entropy from the entire hash code.
    // It's based on Fibonacci hashing from Knuth's Art of Computer
    // Programming volume 3, section 6.4.
    const uint64 version_hash = [&]() -> uint64 {
      if (request.version) {
        return std::hash<int64>()(request.version.value()) *
               0x9E3779B9;  // (sqrt(5) - 1)/2 as a binary fraction.
      } else {
        return 0x9E3779B9;
      }
    }();
    // Using version_hash as the seed here to combine the hashes.
    return Hash64(request.name.data(), request.name.size(), version_hash);
  }
};

DynamicManager::ServingMap::ServingMap()
    : handles_map_(std::unique_ptr<HandlesMap>(new HandlesMap())) {}

std::vector<ServableId> DynamicManager::ServingMap::ListAvailableServableIds() {
  std::vector<ServableId> ids;
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  for (auto iter = handles_map->begin(); iter != handles_map->end();) {
    // We get the iterator where all the values for a particular key ends.
    const auto key_end = handles_map->equal_range(iter->first).second;

    for (; iter != key_end; ++iter) {
      if (iter->first.version) {
        ids.push_back(iter->second->id());
      }
    }
  }
  return ids;
}

Source<std::unique_ptr<Loader>>::AspiredVersionsCallback
DynamicManager::GetAspiredVersionsCallback() {
  return target_impl_->GetAspiredVersionsCallback();
}

Status DynamicManager::ServingMap::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  const auto found_it = handles_map->find(request);
  if (found_it == handles_map->end()) {
    return errors::NotFound(strings::StrCat("Servable not found for request: ",
                                            request.DebugString()));
  }

  // We use the aliasing constructor of shared_ptr here. So even though we
  // are returning a shared_ptr to servable, the ref-counting is happening
  // on the handles_map. This delays the map destruction till the last
  // handle from the previous map is freed, when we are doing handles_map
  // updates.
  untyped_handle->reset(new SharedPtrHandle(
      std::shared_ptr<Loader>(handles_map, found_it->second->loader())));
  return Status::OK();
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
DynamicManager::ServingMap::GetAvailableUntypedServableHandles() const {
  std::map<ServableId, std::unique_ptr<UntypedServableHandle>> result;
  std::shared_ptr<const HandlesMap> handles_map = handles_map_.get();
  for (const auto& handle : *handles_map) {
    const ServableRequest& request = handle.first;
    // If the entry is the one for the latest request, skip it. We would already
    // get it from the entry which has the specific request.
    if (!request.version) {
      continue;
    }
    const ServableId id = {request.name, request.version.value()};
    result.emplace(id, std::unique_ptr<UntypedServableHandle>(
                           new SharedPtrHandle(std::shared_ptr<Loader>(
                               handles_map, handle.second->loader()))));
  }
  return result;
}

void DynamicManager::ServingMap::Update(
    const ManagedMap& managed_map_,
    EventBus<ServableState>* const servable_event_bus) {
  struct CompareRequests {
    bool operator()(const ServableRequest& lhs,
                    const ServableRequest& rhs) const {
      const int strcmp_result = lhs.name.compare(rhs.name);
      if (strcmp_result != 0) {
        return strcmp_result < 0;
      }
      DCHECK(lhs.version);
      DCHECK(rhs.version);
      return lhs.version.value() < rhs.version.value();
    }
  };
  std::multimap<ServableRequest, std::shared_ptr<const LoaderHarness>,
                CompareRequests>
      sorted_managed_map;
  for (const auto& elem : managed_map_) {
    const ServableRequest request = ServableRequest::Specific(
        elem.second->id().name, elem.second->id().version);
    sorted_managed_map.emplace(request, elem.second);
  }

  const std::vector<ServableId> available_servable_ids =
      ListAvailableServableIds();
  const std::unordered_set<ServableId, HashServableId>
      old_available_servable_ids(available_servable_ids.begin(),
                                 available_servable_ids.end());

  const LoaderHarness* newly_available_harness = nullptr;
  std::unique_ptr<HandlesMap> new_handles_map(new HandlesMap());
  for (auto iter = sorted_managed_map.begin(); iter != sorted_managed_map.end();
       ++iter) {
    if (iter->second->state() == LoaderHarness::State::kReady) {
      if (old_available_servable_ids.count(iter->second->id()) == 0) {
        newly_available_harness = iter->second.get();
      }
      new_handles_map->insert(*iter);
      // If this is the last element with this servable name, add it again to
      // the handles_map, marking it as latest.
      const auto next_iter = std::next(iter);
      if (next_iter == sorted_managed_map.end() ||
          next_iter->second->id().name != iter->second->id().name) {
        const ServableRequest latest_request =
            ServableRequest::Latest(iter->second->id().name);
        // We don't set the version to mark it as latest.
        new_handles_map->emplace(latest_request, iter->second);
      }
    }
  }

  // This blocks until the last handle given out by the old handles map is
  // freed.
  handles_map_.Update(std::move(new_handles_map));
  if (newly_available_harness != nullptr && servable_event_bus != nullptr) {
    servable_event_bus->Publish({newly_available_harness->id(),
                                 ServableState::ManagerState::kAvailable,
                                 newly_available_harness->status()});
  }
}

DynamicManager::DynamicManager(Options options)
    : options_(std::move(options)),
      target_impl_(new internal::DynamicManagerTargetImpl(this)) {
  if (options_.manage_state_interval_micros > 0) {
    PeriodicFunction::Options pf_options;
    pf_options.env = options_.env;
    pf_options.thread_name_prefix = "DynamicManager_ManageState_thread";
    manage_state_thread_.reset(
        new PeriodicFunction([this]() { this->ManageState(); },
                             options_.manage_state_interval_micros));
  }
}

DynamicManager::~DynamicManager() {
  // This will wait till the thread is joined.
  manage_state_thread_.reset();

  UnloadAllServables();
}

void DynamicManager::UnloadAllServables() {
  {
    mutex_lock l(managed_map_mu_);
    for (auto it = managed_map_.begin(); it != managed_map_.end(); ++it) {
      if (it->second->state() == LoaderHarness::State::kReady) {
        it->second->StartQuiescing();
        it->second->DoneQuiescing();
        it->second->Unload();
      }
      if (it->second->state() == LoaderHarness::State::kQuiescing) {
        it->second->DoneQuiescing();
        it->second->Unload();
      }
      if (it->second->state() == LoaderHarness::State::kQuiesced) {
        it->second->Unload();
      }
    }
  }
}

std::vector<ServableId> DynamicManager::ListAvailableServableIds() {
  return serving_map_.ListAvailableServableIds();
}

Status DynamicManager::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  return serving_map_.GetUntypedServableHandle(request, untyped_handle);
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
DynamicManager::GetAvailableUntypedServableHandles() const {
  return serving_map_.GetAvailableUntypedServableHandles();
}

void DynamicManager::SetAspiredVersions(
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
    mutex_lock l(managed_map_mu_);

    // We gather all the servable harnesses with the servable_name and
    // 1. Add the current aspired version numbers to a vector and sort it,
    // 2. Set the aspired bool to false for all current servable harnesses which
    // are not aspired.
    const auto range = managed_map_.equal_range(servable_name.ToString());
    std::vector<int64> current_aspired_versions;
    current_aspired_versions.reserve(std::distance(range.first, range.second));
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second->is_aspired()) {
        current_aspired_versions.push_back(it->second->id().version);
      }
      // If this version is not part of the aspired versions.
      if (std::find(next_aspired_versions.begin(), next_aspired_versions.end(),
                    it->second->id().version) == next_aspired_versions.end()) {
        it->second->set_is_aspired(false);
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
        std::unique_ptr<Loader> loader;
        if (version.status().ok()) {
          loader = version.ConsumeDataOrDie();
        }
        std::shared_ptr<LoaderHarness> harness =
            std::make_shared<LoaderHarness>(
                version.id(), std::move(loader),
                LoaderHarness::Options{options_.max_num_load_tries,
                                       options_.load_retry_interval_micros});
        if (!version.status().ok()) {
          LOG(ERROR) << "Version error: " << version.status().ToString();
          harness->Error(version.status());
          PublishOnEventBus({harness->id(), ServableState::ManagerState::kEnd,
                             harness->status()});
        } else {
          managed_map_.emplace(servable_name.ToString(), harness);
          PublishOnEventBus({harness->id(), ServableState::ManagerState::kStart,
                             harness->status()});
        }
      }
    }
  }
}

void DynamicManager::UpdateServingMap() {
  // This blocks until the last handle given out by the old serving map is
  // freed.
  serving_map_.Update(managed_map_, options_.servable_event_bus);
}

DynamicManager::ManagedMap::iterator DynamicManager::FindHarnessInMap(
    const ServableId& id) {
  const auto range = managed_map_.equal_range(id.name);
  for (auto iter = range.first; iter != range.second; ++iter) {
    if (iter->second->id().version == id.version) {
      return iter;
    }
  }
  return managed_map_.end();
}

bool DynamicManager::UnloadQuiesced() {
  for (auto iter = managed_map_.begin(); iter != managed_map_.end();) {
    // We get the iterator where all the values for a particular key ends.
    const auto key_end = managed_map_.equal_range(iter->first).second;

    for (; iter != key_end; ++iter) {
      if (iter->second->state() == LoaderHarness::State::kQuiesced) {
        // At this point, this LoaderHarness should have been removed from the
        // serving map and only be present in the managed map.
        if (!iter->second.unique()) {
          LOG(ERROR) << "Memory leak! LoaderHarness in quiesced state should "
                        "only be present in the managed map."
                     << iter->second->id().DebugString();
          DCHECK(false) << "See previous memory leak error message.";
          return false;
        }
        iter->second->Unload();
        PublishOnEventBus({iter->second->id(),
                           ServableState::ManagerState::kEnd,
                           iter->second->status()});
        // This erase will lead to the LoaderHarness being deleted.
        iter = managed_map_.erase(iter);
        return true;
      }
    }
  }
  return false;
}

// We collect the version policy actions for each servable stream first. Then
// we sort them based on the global policy and pick the first one.
optional<VersionPolicy::ServableAction> DynamicManager::GetNextAction() {
  std::vector<optional<VersionPolicy::ServableAction>> actions;
  for (auto iter = managed_map_.begin(); iter != managed_map_.end();) {
    // We get the iterator where all the values for a particular key ends.
    const auto key_end = managed_map_.equal_range(iter->first).second;

    std::vector<ServableStateSnapshot> state_snapshots;
    state_snapshots.reserve(std::distance(iter, key_end));
    for (; iter != key_end; ++iter) {
      state_snapshots.push_back(iter->second->loader_state_snapshot());
    }
    actions.emplace_back(
        options_.version_policy->GetNextAction(state_snapshots));
  }

  std::sort(actions.begin(), actions.end(), CompareActions());
  const optional<VersionPolicy::ServableAction> next_action =
      !actions.empty() ? actions[0] : nullopt;
  return next_action;
}

void DynamicManager::PerformAction(const VersionPolicy::Action action,
                                   ManagedMap::iterator harness_iter) {
  switch (action) {
    case VersionPolicy::Action::kUnload:
      PublishOnEventBus({harness_iter->second->id(),
                         ServableState::ManagerState::kUnloading,
                         harness_iter->second->status()});
      harness_iter->second->StartQuiescing();
      break;
    case VersionPolicy::Action::kLoad: {
      // TODO(b/27494466): Pass a real value for 'available_resources'.
      const ResourceAllocation available_resources;
      const Status status = harness_iter->second->Load(available_resources);
      PublishOnEventBus({harness_iter->second->id(),
                         ServableState::ManagerState::kLoading,
                         harness_iter->second->status()});
      if (!status.ok()) {
        // The harness has already placed the loader in the kError state, so
        // aside from logging the error for debugging purposes there is nothing
        // to do here from an error-handling perspective.
        LOG(ERROR) << "Servable load failure: " << status;
        PublishOnEventBus({harness_iter->second->id(),
                           ServableState::ManagerState::kEnd,
                           harness_iter->second->status()});
      }
      break;
    }
  }
}

void DynamicManager::ManageState() {
  mutex_lock l(managed_map_mu_);

  if (UnloadQuiesced()) {
    return;
  }

  const optional<VersionPolicy::ServableAction> next_action = GetNextAction();
  if (!next_action) {
    return;
  }

  // We could action validation here.

  auto harness_iter = FindHarnessInMap(next_action.value().id);
  // 'harness_iter' should refer to an element, given that we just used the
  // snapshots from the harness map.
  if (harness_iter == managed_map_.end()) {
    LOG(ERROR) << "Implementation invariant violated; "
                  "DynamicManager::ManageState() exiting early";
    DCHECK(false);
    return;
  }

  PerformAction(next_action.value().action, harness_iter);

  // TODO(b/25357074): Maybe move to its own thread.
  UpdateServingMap();

  // After the update, a servable in quiescing state would have been removed
  // from the serving map, so change the state of that servable in quiescing
  // to quiesced.
  for (const auto& elem : managed_map_) {
    if (elem.second->state() == LoaderHarness::State::kQuiescing) {
      elem.second->DoneQuiescing();
      return;
    }
  }
}

void DynamicManager::PublishOnEventBus(const ServableState& state) {
  if (options_.servable_event_bus != nullptr) {
    options_.servable_event_bus->Publish(state);
  }
}

}  // namespace serving
}  // namespace tensorflow
