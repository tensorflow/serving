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

#include "tensorflow_serving/core/manager_wrapper.h"

namespace tensorflow {
namespace serving {

ManagerWrapper::ManagerWrapper(UniquePtrWithDeps<Manager> wrapped)
    : wrapped_(std::move(wrapped)) {}

std::vector<ServableId> ManagerWrapper::ListAvailableServableIds() const {
  return wrapped_->ListAvailableServableIds();
}

Status ManagerWrapper::GetUntypedServableHandle(
    const ServableRequest& request,
    std::unique_ptr<UntypedServableHandle>* const untyped_handle) {
  return wrapped_->GetUntypedServableHandle(request, untyped_handle);
}

std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
ManagerWrapper::GetAvailableUntypedServableHandles() const {
  return wrapped_->GetAvailableUntypedServableHandles();
}

}  // namespace serving
}  // namespace tensorflow
