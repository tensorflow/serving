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

#ifndef TENSORFLOW_SERVING_UTIL_FILE_PROBING_ENV_H_
#define TENSORFLOW_SERVING_UTIL_FILE_PROBING_ENV_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {

// An interface used to probe the contents of a file system. This allows file
// systems other than those supported by tensorflow::Env to be used.
class FileProbingEnv {
 public:
  virtual ~FileProbingEnv() = default;

  // Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual Status FileExists(const string& fname) = 0;

  // Stores in *children the names of the children of the specified
  // directory. The names are relative to "dir".
  // Original contents of *children are dropped.
  virtual Status GetChildren(const string& dir,
                             std::vector<string>* children) = 0;

  // Returns whether the given path is a directory or not.
  // See tensorflow::Env::IsDirectory() for possible status code.
  virtual Status IsDirectory(const string& fname) = 0;

  // Stores the size of `fname` in `*file_size`.
  virtual Status GetFileSize(const string& fname, uint64* file_size) = 0;
};

// An implementation of FileProbingEnv which delegates the calls to
// tensorflow::Env.
class TensorflowFileProbingEnv : public FileProbingEnv {
 public:
  // 'env' is owned by the caller.
  TensorflowFileProbingEnv(tensorflow::Env* env) : env_(env) {}

  ~TensorflowFileProbingEnv() override = default;

  Status FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* children) override;

  Status IsDirectory(const string& fname) override;

  Status GetFileSize(const string& fname, uint64* file_size) override;

 private:
  // Not owned.
  tensorflow::Env* env_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_FILE_PROBING_ENV_H_
