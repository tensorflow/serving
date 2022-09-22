/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/server_init.h"

#include "tensorflow_serving/model_servers/http_rest_api_handler.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/prediction_service_impl.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.h"

namespace tensorflow {
namespace serving {
namespace init {

Status SetupPlatformConfigMapForTensorFlowImpl(
    const SessionBundleConfig& session_bundle_config,
    PlatformConfigMap& platform_config_map) {
  platform_config_map =
      CreateTensorFlowPlatformConfigMap(session_bundle_config);
  return tensorflow::OkStatus();
}

Status UpdatePlatformConfigMapForTensorFlowImpl(
    PlatformConfigMap& platform_config_map) {
  return tensorflow::OkStatus();
}

std::unique_ptr<HttpRestApiHandlerBase> CreateHttpRestApiHandlerImpl(
    int timeout_in_ms, ServerCore* core) {
  return absl::make_unique<HttpRestApiHandler>(timeout_in_ms, core);
}

std::unique_ptr<PredictionService::Service> CreatePredictionServiceImpl(
    const PredictionServiceOptions& options) {
  return absl::make_unique<PredictionServiceImpl>(options);
}

ABSL_CONST_INIT SetupPlatformConfigMapForTensorFlowFnType
    SetupPlatformConfigMapForTensorFlow =
        SetupPlatformConfigMapForTensorFlowImpl;
ABSL_CONST_INIT UpdatePlatformConfigMapForTensorFlowFnType
    UpdatePlatformConfigMapForTensorFlow =
        UpdatePlatformConfigMapForTensorFlowImpl;
ABSL_CONST_INIT CreateHttpRestApiHandlerFnType CreateHttpRestApiHandler =
    CreateHttpRestApiHandlerImpl;
ABSL_CONST_INIT CreatePredictionServiceFnType CreatePredictionService =
    CreatePredictionServiceImpl;

}  // namespace init
}  // namespace serving
}  // namespace tensorflow
