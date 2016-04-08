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

// Polls the file system, and populates 'version' with the aspired-versions data
// FileSystemStoragePathSource should emit based on what was found.
Status PollFileSystem(const FileSystemStoragePathSourceConfig& config,
                      std::vector<ServableData<StoragePath>>* versions) {
  // First, determine whether the base path exists. This check guarantees that
  // we don't emit an empty aspired-versions list for a non-existent (or
  // transiently unavailable) base-path. (On some platforms, GetChildren()
  // returns an empty list instead of erring if the base path isn't found.)
  if (!Env::Default()->FileExists(config.base_path())) {
    return errors::InvalidArgument("Could not find base path: ",
                                   config.base_path());
  }

  // Retrieve a list of base-path children from the file system.
  std::vector<string> children;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetChildren(config.base_path(), &children));

  // Identify the latest version, among children that can be interpreted as
  // version numbers.
  int latest_version_child = -1;
  int64 latest_version;
  for (int i = 0; i < children.size(); ++i) {
    const string& child = children[i];

    int64 child_version_num;
    if (!strings::safe_strto64(child.c_str(), &child_version_num)) {
      continue;
    }

    if (latest_version_child < 0 || latest_version < child_version_num) {
      latest_version_child = i;
      latest_version = child_version_num;
    }
  }

  // Emit the aspired-versions data.
  if (latest_version_child >= 0) {
    const ServableId servable_id = {config.servable_name(), latest_version};
    const string full_path =
        io::JoinPath(config.base_path(), children[latest_version_child]);
    versions->emplace_back(ServableData<StoragePath>(servable_id, full_path));
  } else {
    LOG(WARNING) << "No servable versions found under base path: "
                 << config.base_path();
  }

  return Status::OK();
}

}  // namespace

Status FileSystemStoragePathSource::Create(
    const FileSystemStoragePathSourceConfig& config,
    std::unique_ptr<FileSystemStoragePathSource>* result) {
  auto raw_result = new FileSystemStoragePathSource();
  raw_result->config_ = config;
  result->reset(raw_result);

  // Determine whether at least one version currently exists.
  if (config.fail_if_zero_versions_at_startup()) {
    std::vector<ServableData<StoragePath>> versions;
    TF_RETURN_IF_ERROR(PollFileSystem(config, &versions));
    if (versions.empty()) {
      return errors::NotFound("Unable to find a numerical version path at: ",
                              config.base_path());
    }
  }

  return Status::OK();
}

void FileSystemStoragePathSource::SetAspiredVersionsCallback(
    AspiredVersionsCallback callback) {
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
  std::vector<ServableData<StoragePath>> versions;
  TF_RETURN_IF_ERROR(PollFileSystem(config_, &versions));
  for (const ServableData<StoragePath>& version : versions) {
    if (version.status().ok()) {
      LOG(INFO) << "Aspiring version for servable " << config_.servable_name()
                << " from path: " << version.DataOrDie();
    }
  }
  aspired_versions_callback_(config_.servable_name(), versions);
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
