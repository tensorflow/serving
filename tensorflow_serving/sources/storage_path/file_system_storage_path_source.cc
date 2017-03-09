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

#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

FileSystemStoragePathSource::~FileSystemStoragePathSource() {
  // Note: Deletion of 'fs_polling_thread_' will block until our underlying
  // thread closure stops. Hence, destruction of this object will not proceed
  // until the thread has terminated.
  fs_polling_thread_.reset();
}

namespace {

// Converts any deprecated usage in 'config' into equivalent non-deprecated use.
// TODO(b/30898016): Eliminate this once the deprecated fields are gone.
FileSystemStoragePathSourceConfig NormalizeConfig(
    const FileSystemStoragePathSourceConfig& config) {
  FileSystemStoragePathSourceConfig normalized = config;
  if (!normalized.servable_name().empty() || !normalized.base_path().empty()) {
    FileSystemStoragePathSourceConfig::ServableToMonitor* servable =
        normalized.add_servables();
    servable->set_servable_name(normalized.servable_name());
    servable->set_base_path(normalized.base_path());
    normalized.clear_servable_name();
    normalized.clear_base_path();
  }
  return normalized;
}

// Returns the names of servables that appear in 'old_config' but not in
// 'new_config'. Assumes both configs are normalized.
std::set<string> GetDeletedServables(
    const FileSystemStoragePathSourceConfig& old_config,
    const FileSystemStoragePathSourceConfig& new_config) {
  std::set<string> new_servables;
  for (const FileSystemStoragePathSourceConfig::ServableToMonitor& servable :
       new_config.servables()) {
    new_servables.insert(servable.servable_name());
  }

  std::set<string> deleted_servables;
  for (const FileSystemStoragePathSourceConfig::ServableToMonitor&
           old_servable : old_config.servables()) {
    if (new_servables.find(old_servable.servable_name()) ==
        new_servables.end()) {
      deleted_servables.insert(old_servable.servable_name());
    }
  }
  return deleted_servables;
}

// Adds a new ServableData for the model version to the vector of versions to
// aspire.
void AspireVersion(
    const FileSystemStoragePathSourceConfig::ServableToMonitor& servable,
    const string& version_relative_path, const int64 version_number,
    std::vector<ServableData<StoragePath>>* versions) {
  const ServableId servable_id = {servable.servable_name(), version_number};
  const string full_path =
      io::JoinPath(servable.base_path(), version_relative_path);
  versions->emplace_back(ServableData<StoragePath>(servable_id, full_path));
}

// Converts the string version path to an integer.
// Returns false if the input is invalid.
bool ParseVersionNumber(const string& version_path, int64* version_number) {
  return strings::safe_strto64(version_path.c_str(), version_number);
}

// Update the servable data to include all the model versions found in the base
// path as aspired versions.
// The argument 'children' represents a list of base-path children from the file
// system.
// Returns true if one or more valid servable version paths are found, otherwise
// returns false.
bool AspireAllVersions(
    const FileSystemStoragePathSourceConfig::ServableToMonitor& servable,
    const std::vector<string>& children,
    std::vector<ServableData<StoragePath>>* versions) {
  bool at_least_one_version_found = false;
  for (const string& child : children) {
    // Identify all the versions, among children that can be interpreted as
    // version numbers.
    int64 version_number;
    if (ParseVersionNumber(child, &version_number)) {
      // Emit all the aspired-versions data.
      AspireVersion(servable, child, version_number, versions);
      at_least_one_version_found = true;
    }
  }

  return at_least_one_version_found;
}

// Update the servable data to include the latest version found in the base path
// as aspired version.
// The argument 'children' represents a list of base-path children from the file
// system.
// Returns true if a valid servable path is found, otherwise returns false.
bool AspireLatestVersion(
    const FileSystemStoragePathSourceConfig::ServableToMonitor& servable,
    const std::vector<string>& children,
    std::vector<ServableData<StoragePath>>* versions) {
  // Identify the latest version among children that can be interpreted as
  // version numbers.
  int latest_version_child_index;
  int64 latest_version_number;
  bool at_least_one_version_found = false;
  for (int i = 0; i < children.size(); ++i) {
    int64 version_number;
    if (!ParseVersionNumber(children[i], &version_number)) {
      continue;
    }

    // Check if this is the largest version number.
    if (!at_least_one_version_found || latest_version_number < version_number) {
      latest_version_child_index = i;
      latest_version_number = version_number;
    }
    at_least_one_version_found = true;
  }

  // Emit the latest aspired version.
  if (at_least_one_version_found) {
    AspireVersion(servable, children[latest_version_child_index],
                  latest_version_number, versions);
  }

  return at_least_one_version_found;
}

// Like PollFileSystemForConfig(), but for a single servable.
Status PollFileSystemForServable(
    const FileSystemStoragePathSourceConfig::ServableToMonitor& servable,
    std::vector<ServableData<StoragePath>>* versions) {
  // First, determine whether the base path exists. This check guarantees that
  // we don't emit an empty aspired-versions list for a non-existent (or
  // transiently unavailable) base-path. (On some platforms, GetChildren()
  // returns an empty list instead of erring if the base path isn't found.)
  if (!Env::Default()->FileExists(servable.base_path()).ok()) {
    return errors::InvalidArgument("Could not find base path ",
                                   servable.base_path(), " for servable ",
                                   servable.servable_name());
  }

  // Retrieve a list of base-path children from the file system.
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetChildren(servable.base_path(), &children));

  // GetChildren() returns all descendants instead for cloud storage like GCS.
  // In such case we should filter out all non-direct descendants.
  std::set<string> real_children;
  for (int i = 0; i < children.size(); ++i) {
    const string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }
  children.clear();
  children.insert(children.begin(), real_children.begin(), real_children.end());

  bool at_least_one_version_found = false;
  switch (servable.version_policy()) {
    case FileSystemStoragePathSourceConfig::LATEST_VERSION:
      at_least_one_version_found =
          AspireLatestVersion(servable, children, versions);
      break;
    case FileSystemStoragePathSourceConfig::ALL_VERSIONS:
      at_least_one_version_found =
          AspireAllVersions(servable, children, versions);
      break;
    default:
      return errors::Internal("Unhandled servable version_policy: ",
                              servable.version_policy());
  }

  if (!at_least_one_version_found) {
    LOG(WARNING) << "No versions of servable " << servable.servable_name()
                 << " found under base path " << servable.base_path();
  }

  return Status::OK();
}

// Polls the file system, and populates 'versions_by_servable_name' with the
// aspired-versions data FileSystemStoragePathSource should emit based on what
// was found, indexed by servable name.
Status PollFileSystemForConfig(
    const FileSystemStoragePathSourceConfig& config,
    std::map<string, std::vector<ServableData<StoragePath>>>*
        versions_by_servable_name) {
  for (const FileSystemStoragePathSourceConfig::ServableToMonitor& servable :
       config.servables()) {
    std::vector<ServableData<StoragePath>> versions;
    TF_RETURN_IF_ERROR(PollFileSystemForServable(servable, &versions));
    versions_by_servable_name->insert(
        {servable.servable_name(), std::move(versions)});
  }
  return Status::OK();
}

// Determines if, for any servables in 'config', the file system doesn't
// currently contain at least one version under its base path.
Status FailIfZeroVersions(const FileSystemStoragePathSourceConfig& config) {
  std::map<string, std::vector<ServableData<StoragePath>>>
      versions_by_servable_name;
  TF_RETURN_IF_ERROR(
      PollFileSystemForConfig(config, &versions_by_servable_name));
  for (const auto& entry : versions_by_servable_name) {
    const string& servable = entry.first;
    const std::vector<ServableData<StoragePath>>& versions = entry.second;
    if (versions.empty()) {
      return errors::NotFound(
          "Unable to find a numerical version path for servable ", servable,
          " at: ", config.base_path());
    }
  }
  return Status::OK();
}

}  // namespace

Status FileSystemStoragePathSource::Create(
    const FileSystemStoragePathSourceConfig& config,
    std::unique_ptr<FileSystemStoragePathSource>* result) {
  result->reset(new FileSystemStoragePathSource());
  return (*result)->UpdateConfig(config);
}

Status FileSystemStoragePathSource::UpdateConfig(
    const FileSystemStoragePathSourceConfig& config) {
  mutex_lock l(mu_);

  if (fs_polling_thread_ != nullptr &&
      config.file_system_poll_wait_seconds() !=
          config_.file_system_poll_wait_seconds()) {
    return errors::InvalidArgument(
        "Changing file_system_poll_wait_seconds is not supported");
  }

  const FileSystemStoragePathSourceConfig normalized_config =
      NormalizeConfig(config);

  if (normalized_config.fail_if_zero_versions_at_startup()) {
    TF_RETURN_IF_ERROR(FailIfZeroVersions(normalized_config));
  }

  if (aspired_versions_callback_) {
    // TODO(b/35997855): Don't just ignore the ::tensorflow::Status object!
    UnaspireServables(GetDeletedServables(config_, normalized_config))
        .IgnoreError();
  }
  config_ = normalized_config;

  return Status::OK();
}

void FileSystemStoragePathSource::SetAspiredVersionsCallback(
    AspiredVersionsCallback callback) {
  mutex_lock l(mu_);

  if (fs_polling_thread_ != nullptr) {
    LOG(ERROR) << "SetAspiredVersionsCallback() called multiple times; "
                  "ignoring this call";
    DCHECK(false);
    return;
  }
  aspired_versions_callback_ = callback;

  if (config_.file_system_poll_wait_seconds() >= 0) {
    // Kick off a thread to poll the file system periodically, and call the
    // callback.
    PeriodicFunction::Options pf_options;
    pf_options.thread_name_prefix =
        "FileSystemStoragePathSource_filesystem_polling_thread";
    fs_polling_thread_.reset(new PeriodicFunction(
        [this] {
          Status status = this->PollFileSystemAndInvokeCallback();
          if (!status.ok()) {
            LOG(ERROR) << "FileSystemStoragePathSource encountered a "
                          "file-system access error: "
                       << status.error_message();
          }
        },
        config_.file_system_poll_wait_seconds() * 1000000, pf_options));
  }
}

Status FileSystemStoragePathSource::PollFileSystemAndInvokeCallback() {
  mutex_lock l(mu_);
  std::map<string, std::vector<ServableData<StoragePath>>>
      versions_by_servable_name;
  TF_RETURN_IF_ERROR(
      PollFileSystemForConfig(config_, &versions_by_servable_name));
  for (const auto& entry : versions_by_servable_name) {
    const string& servable = entry.first;
    const std::vector<ServableData<StoragePath>>& versions = entry.second;
    for (const ServableData<StoragePath>& version : versions) {
      if (version.status().ok()) {
        VLOG(1) << "File-system polling update: Servable:" << version.id()
                << "; Servable path: " << version.DataOrDie()
                << "; Polling frequency: "
                << config_.file_system_poll_wait_seconds();
      }
    }
    aspired_versions_callback_(servable, versions);
  }
  return Status::OK();
}

Status FileSystemStoragePathSource::UnaspireServables(
    const std::set<string>& servable_names) {
  for (const string& servable_name : servable_names) {
    aspired_versions_callback_(servable_name, {});
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
