/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tfrt_get_model_metadata_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tsl/platform/errors.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"

namespace tensorflow {
namespace serving {

absl::Status TFRTGetModelMetadataImpl::GetModelMetadata(
    ServerCore* core, const GetModelMetadataRequest& request,
    GetModelMetadataResponse* response) {
  if (!request.has_model_spec()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Missing ModelSpec");
  }
  return GetModelMetadataWithModelSpec(core, request.model_spec(), request,
                                       response);
}

absl::Status TFRTGetModelMetadataImpl::GetModelMetadataWithModelSpec(
    ServerCore* core, const ModelSpec& model_spec,
    const GetModelMetadataRequest& request,
    GetModelMetadataResponse* response) {
  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &servable));
  TF_RETURN_IF_ERROR(servable->GetModelMetadata(request, response));
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
