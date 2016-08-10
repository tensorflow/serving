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

#include "tensorflow_serving/model_servers/test_util/server_core_test_util.h"

namespace tensorflow {
namespace serving {
namespace test_util {

std::vector<ServableId> ServerCoreTestAccess::ListAvailableServableIds() const {
  return core_->ListAvailableServableIds();
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
