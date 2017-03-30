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

#include "tensorflow_serving/servables/tensorflow/regression_service.h"

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow_serving/apis/regressor.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/regressor.h"

namespace tensorflow {
namespace serving {

Status TensorflowRegressionServiceImpl::Regress(
    const RunOptions& run_options, ServerCore* core,
    const RegressionRequest& request, RegressionResponse* response) {
  TRACELITERAL("TensorflowRegressionServiceImpl::Regress");
  // Verify Request Metadata and create a ServableRequest
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }

  ServableHandle<SavedModelBundle> saved_model_bundle;
  TF_RETURN_IF_ERROR(
      core->GetServableHandle(request.model_spec(), &saved_model_bundle));
  SignatureDef signature;
  TF_RETURN_IF_ERROR(GetRegressionSignatureDef(
      request.model_spec(), saved_model_bundle->meta_graph_def, &signature));

  std::unique_ptr<RegressorInterface> regressor_interface;
  TF_RETURN_IF_ERROR(CreateFlyweightTensorFlowRegressor(
      run_options, saved_model_bundle->session.get(), &signature,
      &regressor_interface));
  // Run regression
  return regressor_interface->Regress(request, response->mutable_result());
}

}  // namespace serving
}  // namespace tensorflow
