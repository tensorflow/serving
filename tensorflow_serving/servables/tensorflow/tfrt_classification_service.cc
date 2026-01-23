/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tfrt_classification_service.h"

#include <memory>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow_serving/apis/classifier.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_classifier.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

absl::Status TFRTClassificationServiceImpl::Classify(
    const Servable::RunOptions& run_options, ServerCore* core,
    const ClassificationRequest& request, ClassificationResponse* response) {
  // Verify Request Metadata and create a ServableRequest
  if (!request.has_model_spec()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Missing ModelSpec");
  }

  return ClassifyWithModelSpec(run_options, core, request.model_spec(), request,
                               response);
}

absl::Status TFRTClassificationServiceImpl::ClassifyWithModelSpec(
    const Servable::RunOptions& run_options, ServerCore* core,
    const ModelSpec& model_spec, const ClassificationRequest& request,
    ClassificationResponse* response) {
  TRACELITERAL("TFRTClassificationServiceImpl::ClassifyWithModelSpec");

  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &servable));
  return servable->Classify(run_options, request, response);
}

}  // namespace serving
}  // namespace tensorflow
