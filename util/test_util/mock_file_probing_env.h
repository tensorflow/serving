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

#ifndef TENSORFLOW_SERVING_UTIL_TEST_UTIL_MOCK_FILE_PROBING_ENV_H_
#define TENSORFLOW_SERVING_UTIL_TEST_UTIL_MOCK_FILE_PROBING_ENV_H_

#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockFileProbingEnv : public FileProbingEnv {
 public:
  MOCK_METHOD1(FileExists, Status(const string& fname));
  MOCK_METHOD2(GetChildren,
               Status(const string& fname, std::vector<string>* children));
  MOCK_METHOD1(IsDirectory, Status(const string& fname));
  MOCK_METHOD2(GetFileSize, Status(const string& fname, uint64* file_size));
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_TEST_UTIL_MOCK_FILE_PROBING_ENV_H_
