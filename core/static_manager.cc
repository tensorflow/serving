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

#include "tensorflow_serving/core/static_manager.h"

namespace tensorflow {
namespace serving {

StaticManagerBuilder::StaticManagerBuilder() {
  BasicManager::Options basic_manager_options;
  // We don't want multithreading.
  basic_manager_options.num_load_threads = 0;
  basic_manager_options.num_unload_threads = 0;
  const Status basic_manager_status =
      BasicManager::Create(std::move(basic_manager_options), &basic_manager_);
  if (!basic_manager_status.ok()) {
    LOG(ERROR) << "Error creating BasicManager: " << health_;
    health_ = basic_manager_status;
  }
}

std::unique_ptr<Manager> StaticManagerBuilder::Build() {
  if (!health_.ok()) {
    LOG(ERROR) << health_;
    return nullptr;
  }

  // If Build() is called again, we'll produce the following error.
  health_ = errors::FailedPrecondition(
      "Build() already called on this StaticManagerBuilder.");

  return std::move(basic_manager_);
}

}  // namespace serving
}  // namespace tensorflow
