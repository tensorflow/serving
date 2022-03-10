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

Status SetupPlatformConfigMapImpl(const SessionBundleConfig&,
                                  PlatformConfigMap&);
Status SetupPlatformConfigMapFromConfigFileImpl(const string&,
                                                PlatformConfigMap&);
std::unique_ptr<HttpRestApiHandlerBase> CreateHttpRestApiHandlerImpl(
    int, ServerCore*);
std::unique_ptr<PredictionService::Service> CreatePredictionServiceImpl(
    const PredictionServiceOptions&);

// Setup the 'TensorFlow' PlatformConfigMap from the specified
// SessionBundleConfig.
extern SetupPlatformConfigMapForTensorFlowFnType
    SetupPlatformConfigMapForTensorFlow;
// If the PlatformConfigMap contains the config for the 'TensorFlow' platform,
// update the PlatformConfigMap when necessary.
extern UpdatePlatformConfigMapForTensorFlowFnType
    UpdatePlatformConfigMapForTensorFlow;
// Create an HttpRestApiHandler object that handles HTTP/REST request APIs for
// serving.
extern CreateHttpRestApiHandlerFnType CreateHttpRestApiHandler;
// Create a PredictionService object that handles gRPC request APIs for serving.
extern CreatePredictionServiceFnType CreatePredictionService;

}  // namespace init
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_INIT_H_
