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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_DYNAMIC_MANAGER_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_DYNAMIC_MANAGER_TEST_UTIL_H_

#include "tensorflow_serving/core/dynamic_manager.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// A test utility that provides access to private DynamicManager members.
class DynamicManagerTestAccess {
 public:
  explicit DynamicManagerTestAccess(DynamicManager* manager);

  // Invokes ManageState() on the manager.
  void RunManageState();

 private:
  DynamicManager* const manager_;

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicManagerTestAccess);
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_DYNAMIC_MANAGER_TEST_UTIL_H_
