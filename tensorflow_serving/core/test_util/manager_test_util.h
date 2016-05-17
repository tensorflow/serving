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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_

#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/caching_manager.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A test utility that provides access to private AspiredVersionsManager
// members.
class AspiredVersionsManagerTestAccess {
 public:
  explicit AspiredVersionsManagerTestAccess(AspiredVersionsManager* manager);

  // Invokes ManageState() on the manager.
  void RunManageState();

 private:
  AspiredVersionsManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(AspiredVersionsManagerTestAccess);
};

// A test utility that provides access to private CachingManager members.
class CachingManagerTestAccess {
 public:
  explicit CachingManagerTestAccess(CachingManager* manager)
      : manager_(manager) {}

  // Returns the size of the load-mutex map that stores the mutex reference per
  // servable-id requested for load.
  int64 GetLoadMutexMapSize() const;

 private:
  CachingManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(CachingManagerTestAccess);
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_MANAGER_TEST_UTIL_H_
