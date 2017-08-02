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

#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {

namespace {

const string kSignatureDef = "signature_def";

Status ValidateGetModelMetadataRequest(const GetModelMetadataRequest& request) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  if (request.metadata_field_size() == 0) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "GetModelMetadataRequest must specify at least one metadata_field");
  }
  for (const auto& metadata_field : request.metadata_field()) {
    if (metadata_field != kSignatureDef) {
      return tensorflow::errors::InvalidArgument(
          "Metadata field %s is not supported", metadata_field);
    }
  }
  return tensorflow::Status::OK();
}

Status SavedModelGetSignatureDef(ServerCore* core,
                                 const GetModelMetadataRequest& request,
                                 GetModelMetadataResponse* response) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  SignatureDefMap signature_def_map;
  for (const auto& signature : bundle->meta_graph_def.signature_def()) {
    (*signature_def_map.mutable_signature_def())[signature.first] =
        signature.second;
  }
  auto model_spec = response->mutable_model_spec();
  model_spec->set_name(bundle.id().name);
  model_spec->mutable_version()->set_value(bundle.id().version);

  (*response->mutable_metadata())[kSignatureDef].PackFrom(signature_def_map);
  return tensorflow::Status::OK();
}

}  // namespace

Status GetModelMetadataImpl::GetModelMetadata(
    ServerCore* core, const GetModelMetadataRequest& request,
    GetModelMetadataResponse* response) {
  TF_RETURN_IF_ERROR(ValidateGetModelMetadataRequest(request));
  for (const auto& metadata_field : request.metadata_field()) {
    if (metadata_field == kSignatureDef) {
      TF_RETURN_IF_ERROR(SavedModelGetSignatureDef(core, request, response));
    } else {
      return tensorflow::errors::InvalidArgument(
          "MetadataField %s is not supported", metadata_field);
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
