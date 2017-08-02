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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOADER_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOADER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A fake loader to be used in tests.
//
// Useful for -
// 1. Erroring out on load.
// 2. Counting the total number of fake loaders alive.
// 3. Detecting if deletion of the loader happened on a particular thread.
//
// This class is thread-safe.
class FakeLoader : public ResourceUnsafeLoader {
 public:
  explicit FakeLoader(int64 servable, const Status load_status = Status::OK());

  ~FakeLoader() override;

  // The status returned during load.
  Status load_status();

  Status Load() override;

  void Unload() override;

  AnyPtr servable() override;

  static int num_fake_loaders();

  // Returns true if a loader was deleted in this thread.
  static bool was_deleted_in_this_thread();

 private:
  // Used to detect the thread in which deletion of a loader occurs.
  static thread_local bool was_deleted_in_this_thread_;

  // Counts the number of FakeLoader objects alive.
  static int num_fake_loaders_ GUARDED_BY(num_fake_loaders_mu_);
  static mutex num_fake_loaders_mu_;

  // The servable returned from this loader.
  //
  // Don't make const or you'll have to change the handle type to 'const int64'.
  int64 servable_;
  // The status returned during load.
  const Status load_status_;
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOADER_H_
