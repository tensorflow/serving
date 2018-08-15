/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_MODEL_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_MODEL_SERVICE_IMPL_H_

#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tensorflow_serving/apis/model_management.pb.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"
#include "tensorflow_serving/apis/model_service.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

class ModelServiceImpl final : public ModelService::Service {
 public:
  explicit ModelServiceImpl(ServerCore *core) : core_(core) {}

  ::grpc::Status GetModelStatus(::grpc::ServerContext *context,
                                const GetModelStatusRequest *request,
                                GetModelStatusResponse *response) override;

  ::grpc::Status HandleReloadConfigRequest(::grpc::ServerContext *context,
                                           const ReloadConfigRequest *request,
                                           ReloadConfigResponse *response);

 private:
  ServerCore *core_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_MODEL_SERVICE_IMPL_H_
