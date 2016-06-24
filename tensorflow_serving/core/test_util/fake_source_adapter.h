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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_SOURCE_ADAPTER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/storage_path.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A fake source adapter that generates string servables from paths.
// When a suffix is provided, it is concatenated with a "/" to the path.
// Example:
// If path = "/a/simple/path" and suffix = "foo", the servable string is
// "a/simple/path/foo".
//
// The constructor may also take a callback to be invoked upon destruction. The
// suffix provided to the source-adapter during construction is passed to the
// string argument of the callback when it is invoked.
class FakeSourceAdapter
    : public SimpleLoaderSourceAdapter<StoragePath, string> {
 public:
  FakeSourceAdapter(const string& suffix = "",
                    std::function<void(const string&)> call_on_destruct = {});

  ~FakeSourceAdapter() override;

  // Returns a function to create a fake source adapter.
  static std::function<Status(
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
  GetCreator();

 private:
  const string suffix_;
  std::function<void(const string&)> call_on_destruct_;
  TF_DISALLOW_COPY_AND_ASSIGN(FakeSourceAdapter);
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_SOURCE_ADAPTER_H_
