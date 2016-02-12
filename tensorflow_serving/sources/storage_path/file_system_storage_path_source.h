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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"
#include "tensorflow_serving/util/periodic_function.h"

namespace tensorflow {
namespace serving {
namespace internal {
class FileSystemStoragePathSourceTestAccess;
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

namespace tensorflow {
namespace serving {

// A storage path source that monitors a file-system path associated with a
// servable, using a thread that polls the file-system periodically. Upon each
// poll, it identifies children of a base path whose name is a number (e.g. 123)
// and returns the path corresponding to the largest number.
// For example, if the base path is /foo/bar, and a poll reveals child paths
// /foo/bar/baz, /foo/bar/123 and /foo/bar/456, the aspired-versions callback is
// called with {{456, "/foo/bar/456"}}.
// If, at any time, the base path is found to contain no numerical children, the
// aspired-versions callback is called with an empty versions list.
class FileSystemStoragePathSource : public Source<StoragePath> {
 public:
  static Status Create(const FileSystemStoragePathSourceConfig& config,
                       std::unique_ptr<FileSystemStoragePathSource>* result);
  ~FileSystemStoragePathSource() override;

  void SetAspiredVersionsCallback(AspiredVersionsCallback callback) override;

 private:
  friend class internal::FileSystemStoragePathSourceTestAccess;

  FileSystemStoragePathSource() = default;

  // Polls the file system and identify numerical children of the base path.
  // If zero such children are found, invokes 'aspired_versions_callback_' with
  // an empty versions list. If one or more such children are found, invokes
  // 'aspired_versions_callback_' with a singleton list containing the largest
  // such child.
  Status PollFileSystemAndInvokeCallback();

  FileSystemStoragePathSourceConfig config_;

  AspiredVersionsCallback aspired_versions_callback_;

  // A thread that periodically calls PollFileSystemAndInvokeCallback().
  std::unique_ptr<PeriodicFunction> fs_polling_thread_;

  TF_DISALLOW_COPY_AND_ASSIGN(FileSystemStoragePathSource);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SOURCES_STORAGE_PATH_FILE_SYSTEM_STORAGE_PATH_SOURCE_H_
