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
#include "tensorflow_serving/experimental/tensorflow/ops/remote_predict/kernels/prediction_service_grpc.h"

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/time/clock.h"

using namespace tensorflow;  // NOLINT(build/namespaces)
namespace tensorflow {
namespace serving {
namespace {

absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return absl::Status();
  }
  return absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                      s.error_message());
}

}  // namespace

PredictionServiceGrpc::PredictionServiceGrpc(
    const std::string& target_address) {
  // TODO(b/159739577): Set security channel from incoming rpc request.
  auto channel = ::grpc::CreateChannel(target_address,
                                       ::grpc::InsecureChannelCredentials());
  stub_ = tensorflow::serving::PredictionService::NewStub(channel);
}

StatusOr<::grpc::ClientContext*> PredictionServiceGrpc::CreateRpc(
    absl::Duration max_rpc_deadline) {
  ::grpc::ClientContext* rpc = new ::grpc::ClientContext();
  // TODO(b/159739577): Set deadline as the min value between
  // the incoming rpc deadline and max_rpc_deadline_millis.
  rpc->set_deadline(std::chrono::system_clock::now() +
                    absl::ToChronoSeconds(max_rpc_deadline));
  return rpc;
}

void PredictionServiceGrpc::Predict(
    ::grpc::ClientContext* rpc, PredictRequest* request,
    PredictResponse* response,
    std::function<void(absl::Status status)> callback) {
  std::function<void(::grpc::Status)> wrapped_callback =
      [callback](::grpc::Status status) { callback(FromGrpcStatus(status)); };

  stub_->experimental_async()->Predict(rpc, request, response,
                                       wrapped_callback);
}

}  // namespace serving
}  // namespace tensorflow
