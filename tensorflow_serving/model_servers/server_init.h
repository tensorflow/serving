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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_INIT_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_INIT_H_

#include <string>

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/http_rest_api_handler_base.h"
#include "tensorflow_serving/model_servers/prediction_service_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace init {

using SetupPlatformConfigMapForTensorFlowFnType =
    Status (*)(const SessionBundleConfig&, PlatformConfigMap&);
using UpdatePlatformConfigMapForTensorFlowFnType =
    Status (*)(PlatformConfigMap&);
using CreateHttpRestApiHandlerFnType =
    std::unique_ptr<HttpRestApiHandlerBase> (*)(int, ServerCore*);
using CreatePredictionServiceFnType =
    std::unique_ptr<PredictionService::Service> (*)(
        const PredictionServiceOptions&);

Status SetupPlatformConfigMapForTensorFlowImpl(const SessionBundleConfig&,
                                               PlatformConfigMap&);
Status UpdatePlatformConfigMapForTensorFlowImpl(PlatformConfigMap&);
std::unique_ptr<HttpRestApiHandlerBase> CreateHttpRestApiHandlerImpl(
    int, ServerCore*);
std::unique_ptr<PredictionService::Service> CreatePredictionServiceImpl(
    const PredictionServiceOptions&);

// Register the tensorflow serving functions.
class TensorflowServingFunctionRegistration {
 public:
  virtual ~TensorflowServingFunctionRegistration() = default;

  // Get the registry singleton.
  static TensorflowServingFunctionRegistration* GetRegistry() {
    static auto* registration = new TensorflowServingFunctionRegistration();
    return registration;
  }

  // The tensorflow serving function registration. For TFRT, the TFRT
  // registration will overwrite the Tensorflow registration.
  void Register(
      absl::string_view type,
      SetupPlatformConfigMapForTensorFlowFnType setup_platform_config_map_func,
      UpdatePlatformConfigMapForTensorFlowFnType
          update_platform_config_map_func,
      CreateHttpRestApiHandlerFnType create_http_rest_api_handler_func,
      CreatePredictionServiceFnType create_prediction_service_func);

  bool IsRegistered() const { return !registration_type_.empty(); }

  SetupPlatformConfigMapForTensorFlowFnType GetSetupPlatformConfigMap() const {
    return setup_platform_config_map_;
  }

  UpdatePlatformConfigMapForTensorFlowFnType GetUpdatePlatformConfigMap()
      const {
    return update_platform_config_map_;
  }

  CreateHttpRestApiHandlerFnType GetCreateHttpRestApiHandler() const {
    return create_http_rest_api_handler_;
  }

  CreatePredictionServiceFnType GetCreatePredictionService() const {
    return create_prediction_service_;
  }

 private:
  TensorflowServingFunctionRegistration() {
    Register("tensorflow", init::SetupPlatformConfigMapForTensorFlowImpl,
             init::UpdatePlatformConfigMapForTensorFlowImpl,
             init::CreateHttpRestApiHandlerImpl,
             init::CreatePredictionServiceImpl);
  }

  // The registration type, indicating the platform, e.g. tensorflow, tfrt.
  std::string registration_type_ = "";

  // Setup the 'TensorFlow' PlatformConfigMap from the specified
  // SessionBundleConfig.
  SetupPlatformConfigMapForTensorFlowFnType setup_platform_config_map_;
  // If the PlatformConfigMap contains the config for the 'TensorFlow'
  // platform, update the PlatformConfigMap when necessary.
  UpdatePlatformConfigMapForTensorFlowFnType update_platform_config_map_;
  // Create an HttpRestApiHandler object that handles HTTP/REST request APIs
  // for serving.
  CreateHttpRestApiHandlerFnType create_http_rest_api_handler_;
  // Create a PredictionService object that handles gRPC request APIs for
  // serving.
  CreatePredictionServiceFnType create_prediction_service_;
};

}  // namespace init
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_INIT_H_
