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

#ifndef TENSORFLOW_SERVING_SOURCES_STORAGE_PATH_FILE_SYSTEM_STORAGE_PATH_SOURCE_H_
#define TENSORFLOW_SERVING_SOURCES_STORAGE_PATH_FILE_SYSTEM_STORAGE_PATH_SOURCE_H_

#include <memory>

#include "tensorflow/contrib/batching/util/periodic_function.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"

namespace tensorflow {
namespace serving {
namespace internal {
class FileSystemStoragePathSourceTestAccess;
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

namespace tensorflow {
namespace serving {

// A storage path source that aspires versions for a given set of servables. For
// each servable, it monitors a given file-system base path. It identifies base-
// path children whose name is a number (e.g. 123) and emits the path
// corresponding to the largest number as the servable's single aspired version.
// (To do the file-system monitoring, it uses a background thread that polls the
// file system periodically.)
//
// For example, if a configured servable's base path is /foo/bar, and a file-
// system poll reveals child paths /foo/bar/baz, /foo/bar/123 and /foo/bar/456,
// the aspired-versions callback is called with {{456, "/foo/bar/456"}}. If, at
// any time, the base path is found to contain no numerical children, the
// aspired-versions callback is called with an empty versions list.
//
// The configured set of servables to monitor can be updated at any time by
// calling UpdateConfig(). If any servables were present in the old config but
// not in the new one, the source will immediately aspire zero versions for that
// servable (causing it to be unloaded in the Manager that ultimately consumes
// the aspired-versions calls).
class FileSystemStoragePathSource : public Source<StoragePath> {
 public:
  static Status Create(const FileSystemStoragePathSourceConfig& config,
                       std::unique_ptr<FileSystemStoragePathSource>* result);
  ~FileSystemStoragePathSource() override;

  // Supplies a new config to use. The set of servables to monitor can be
  // changed at any time (see class comment for more information), but it is
  // illegal to change the file-system polling period once
  // SetAspiredVersionsCallback() has been called.
  Status UpdateConfig(const FileSystemStoragePathSourceConfig& config);

  void SetAspiredVersionsCallback(AspiredVersionsCallback callback) override;

  FileSystemStoragePathSourceConfig config() const {
    mutex_lock l(mu_);
    return config_;
  }

 private:
  friend class internal::FileSystemStoragePathSourceTestAccess;

  FileSystemStoragePathSource() = default;

  // Polls the file system and identify numerical children of the base path.
  // If zero such children are found, invokes 'aspired_versions_callback_' with
  // an empty versions list. If one or more such children are found, invokes
  // 'aspired_versions_callback_' with a singleton list containing the largest
  // such child.
  Status PollFileSystemAndInvokeCallback();

  // Sends empty aspired-versions lists for each servable in 'servable_names'.
  Status UnaspireServables(const std::set<string>& servable_names)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;

  FileSystemStoragePathSourceConfig config_ GUARDED_BY(mu_);

  AspiredVersionsCallback aspired_versions_callback_ GUARDED_BY(mu_);

  // A thread that periodically calls PollFileSystemAndInvokeCallback().
  std::unique_ptr<PeriodicFunction> fs_polling_thread_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FileSystemStoragePathSource);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SOURCES_STORAGE_PATH_FILE_SYSTEM_STORAGE_PATH_SOURCE_H_
