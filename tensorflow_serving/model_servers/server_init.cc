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

#include <memory>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow_serving/model_servers/http_rest_api_handler.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/prediction_service_impl.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.h"

namespace tensorflow {
namespace serving {
namespace init {

absl::Status SetupPlatformConfigMapForTensorFlowImpl(
    const SessionBundleConfig& session_bundle_config,
    PlatformConfigMap& platform_config_map) {
  platform_config_map =
      CreateTensorFlowPlatformConfigMap(session_bundle_config);
  return absl::OkStatus();
}

absl::Status UpdatePlatformConfigMapForTensorFlowImpl(
    PlatformConfigMap& platform_config_map) {
  return absl::OkStatus();
}

std::unique_ptr<HttpRestApiHandlerBase> CreateHttpRestApiHandlerImpl(
    int timeout_in_ms, ServerCore* core) {
  return absl::make_unique<HttpRestApiHandler>(timeout_in_ms, core);
}

std::unique_ptr<PredictionService::Service> CreatePredictionServiceImpl(
    const PredictionServiceOptions& options) {
  return absl::make_unique<PredictionServiceImpl>(options);
}

void TensorflowServingFunctionRegistration::Register(
    absl::string_view type,
    SetupPlatformConfigMapForTensorFlowFnType setup_platform_config_map_func,
    UpdatePlatformConfigMapForTensorFlowFnType update_platform_config_map_func,
    CreateHttpRestApiHandlerFnType create_http_rest_api_handler_func,
    CreatePredictionServiceFnType create_prediction_service_func) {
  VLOG(1) << "Registering serving functions for " << type;
  registration_type_ = type;
  setup_platform_config_map_ = setup_platform_config_map_func;
  update_platform_config_map_ = update_platform_config_map_func;
  create_http_rest_api_handler_ = create_http_rest_api_handler_func;
  create_prediction_service_ = create_prediction_service_func;
}

}  // namespace init
}  // namespace serving
}  // namespace tensorflow
