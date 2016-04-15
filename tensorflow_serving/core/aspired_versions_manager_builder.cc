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

#include "tensorflow_serving/core/aspired_versions_manager_builder.h"

#include "tensorflow_serving/core/manager_wrapper.h"

namespace tensorflow {
namespace serving {

Status AspiredVersionsManagerBuilder::Create(
    Options options, std::unique_ptr<AspiredVersionsManagerBuilder>* builder) {
  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  TF_RETURN_IF_ERROR(AspiredVersionsManager::Create(std::move(options),
                                                    &aspired_versions_manager));
  builder->reset(
      new AspiredVersionsManagerBuilder(std::move(aspired_versions_manager)));
  return Status::OK();
}

AspiredVersionsManagerBuilder::AspiredVersionsManagerBuilder(
    std::unique_ptr<AspiredVersionsManager> manager)
    : aspired_versions_manager_(manager.get()) {
  manager_with_sources_.SetOwned(std::move(manager));
}

std::unique_ptr<Manager> AspiredVersionsManagerBuilder::Build() {
  return std::unique_ptr<Manager>(
      new ManagerWrapper(std::move(manager_with_sources_)));
}

}  // namespace serving
}  // namespace tensorflow
