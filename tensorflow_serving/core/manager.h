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

#ifndef TENSORFLOW_SERVING_CORE_MANAGER_H_
#define TENSORFLOW_SERVING_CORE_MANAGER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

/// A query for a specific loaded servable object. The request can either
/// specify a specific version number, or simply opt to use the earliest or
/// latest loaded version.
struct ServableRequest {
  // Initialization factories, for convenience and readability.
  static ServableRequest Specific(const string& name, const int64_t version);
  static ServableRequest Earliest(const string& name);
  static ServableRequest Latest(const string& name);
  static ServableRequest FromId(const ServableId& id);

  // The name of a servable stream.
  string name;

  // An optional specific version number to use.
  absl::optional<int64_t> version;

  // How to choose a version number automatically, if 'version' is left unset.
  enum class AutoVersionPolicy {
    // Choose the loaded version with the smallest version number.
    kEarliest,

    // Choose the loaded version with the largest version number.
    kLatest,
  };
  AutoVersionPolicy auto_version_policy = AutoVersionPolicy::kLatest;

  // Emits a string representation; for logging and debugging use only.
  string DebugString() const;

  ////////
  // Legacy constructors. Do not use in new code.
  ServableRequest() = default;
  ServableRequest(const string& name_in,
                  const absl::optional<int64_t>& version_in)
      : name(name_in),
        version(version_in),
        auto_version_policy(AutoVersionPolicy::kLatest) {}
};

/// Manager is responsible for loading, unloading, lookup and lifetime
/// management of all Servable objects via their Loaders.
class Manager {
 public:
  virtual ~Manager() = default;

  /// Gets a list of all available servable ids, i.e. each of these can
  /// be retrieved using GetServableHandle.
  virtual std::vector<ServableId> ListAvailableServableIds() const = 0;

  /// Returns a map of all the currently available servables of a particular
  /// type T. The map is from the servable's id to its corresponding handle.
  ///
  /// IMPORTANT: The caller should not hold onto the handles for a long time,
  /// because holding them will delay servable loading and unloading.
  template <typename T>
  std::map<ServableId, ServableHandle<T>> GetAvailableServableHandles() const;

  /// Returns a ServableHandle given a ServableRequest. Returns error if no such
  /// Servable is available -- e.g. not yet loaded, has been quiesced/unloaded,
  /// etc. Callers may assume that an OK status indicates a non-null handle.
  ///
  /// IMPORTANT: The caller should not hold onto the handles for a long time,
  /// because holding them will delay servable loading and unloading.
  template <typename T>
  Status GetServableHandle(const ServableRequest& request,
                           ServableHandle<T>* const handle);

 private:
  friend class ManagerWrapper;

  // Returns an UntypedServableHandle given a ServableRequest.
  // Returns error if no such Servable is available -- e.g. not yet loaded, has
  // been quiesced/unloaded, etc.
  virtual Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* untyped_handle) = 0;

  // Returns a map of all the available servable ids to their corresponding
  // UntypedServableHandles.
  virtual std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const = 0;
};

//
//  Implementation details follow. API users need not read.
//

inline ServableRequest ServableRequest::Specific(const string& name,
                                                 const int64_t version) {
  ServableRequest request;
  request.name = name;
  request.version = version;
  return request;
}

inline ServableRequest ServableRequest::Earliest(const string& name) {
  ServableRequest request;
  request.name = name;
  request.auto_version_policy = AutoVersionPolicy::kEarliest;
  return request;
}

inline ServableRequest ServableRequest::Latest(const string& name) {
  ServableRequest request;
  request.name = name;
  request.auto_version_policy = AutoVersionPolicy::kLatest;
  return request;
}

inline ServableRequest ServableRequest::FromId(const ServableId& id) {
  DCHECK_GE(id.version, 0);
  return Specific(id.name, id.version);
}

inline string ServableRequest::DebugString() const {
  if (version) {
    return strings::StrCat("Specific(", name, ", ", version.value(), ")");
  } else {
    switch (auto_version_policy) {
      case AutoVersionPolicy::kEarliest:
        return strings::StrCat("Earliest(", name, ")");
      case AutoVersionPolicy::kLatest:
        return strings::StrCat("Latest(", name, ")");
    }
  }
}

template <typename T>
Status Manager::GetServableHandle(const ServableRequest& request,
                                  ServableHandle<T>* const handle) {
  std::unique_ptr<UntypedServableHandle> untyped_handle;
  TF_RETURN_IF_ERROR(GetUntypedServableHandle(request, &untyped_handle));
  if (untyped_handle == nullptr) {
    return errors::Internal("Manager returned a null handle with OK status.");
  }
  *handle = ServableHandle<T>(std::move(untyped_handle));
  if (handle->get() == nullptr) {
    return errors::InvalidArgument(
        "Servable type doesn't match the asked for type.");
  }
  return Status();
}

template <typename T>
std::map<ServableId, ServableHandle<T>> Manager::GetAvailableServableHandles()
    const {
  std::map<ServableId, ServableHandle<T>> id_and_handles;
  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
      id_and_untyped_handles = GetAvailableUntypedServableHandles();
  for (auto& id_and_untyped_handle : id_and_untyped_handles) {
    auto handle = ServableHandle<T>(std::move(id_and_untyped_handle.second));
    if (handle.get() != nullptr) {
      id_and_handles.emplace(id_and_untyped_handle.first, std::move(handle));
    }
  }
  return id_and_handles;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_MANAGER_H_
