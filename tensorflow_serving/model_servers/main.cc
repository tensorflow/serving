/* Copyright 2016 Google Inc. All Rights Reserved.

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

// gRPC server implementation of
// tensorflow_serving/apis/prediction_service.proto.
//
// It bring up a standard server to serve a single TensorFlow model using
// command line flags, or multiple models via config file.
//
// ModelServer prioritizes easy invocation over flexibility,
// and thus serves a statically configured set of models. New versions of these
// models will be loaded and managed over time using the EagerLoadPolicy at:
//     tensorflow_serving/core/eager_load_policy.h.
// by AspiredVersionsManager at:
//     tensorflow_serving/core/aspired_versions_manager.h
//
// ModelServer has inter-request batching support built-in, by using the
// BatchingSession at:
//     tensorflow_serving/batching/batching_session.h
//
// To serve a single model, run with:
//     $path_to_binary/tensorflow_model_server \
//     --model_base_path=[/tmp/my_model | gs://gcs_address] \
// To specify model name (default "default"): --model_name=my_name
// To specify port (default 8500): --port=my_port
// To enable batching (default disabled): --enable_batching
// To log on stderr (default disabled): --alsologtostderr

#include <unistd.h>
#include <iostream>
#include <memory>
#include <utility>

#include "google/protobuf/wrappers.pb.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/apis/prediction_service.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"

using tensorflow::serving::BatchingParameters;
using tensorflow::serving::EventBus;
using tensorflow::serving::Loader;
using tensorflow::serving::ModelServerConfig;
using tensorflow::serving::ServableState;
using tensorflow::serving::ServableStateMonitor;
using tensorflow::serving::ServerCore;
using tensorflow::serving::SessionBundleSourceAdapter;
using tensorflow::serving::SessionBundleSourceAdapterConfig;
using tensorflow::serving::Target;
using tensorflow::serving::TensorflowPredictImpl;
using tensorflow::serving::UniquePtrWithDeps;
using tensorflow::string;

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

namespace {

constexpr char kTensorFlowModelType[] = "tensorflow";

tensorflow::Status CreateSourceAdapter(
    const SessionBundleSourceAdapterConfig& config, const string& model_type,
    std::unique_ptr<ServerCore::ModelServerSourceAdapter>* adapter) {
  CHECK(model_type == kTensorFlowModelType)  // Crash ok
      << "ModelServer supports only TensorFlow model.";
  std::unique_ptr<SessionBundleSourceAdapter> typed_adapter;
  TF_RETURN_IF_ERROR(
      SessionBundleSourceAdapter::Create(config, &typed_adapter));
  *adapter = std::move(typed_adapter);
  return tensorflow::Status::OK();
}

tensorflow::Status CreateServableStateMonitor(
    EventBus<ServableState>* event_bus,
    std::unique_ptr<ServableStateMonitor>* monitor) {
  monitor->reset(new ServableStateMonitor(event_bus));
  return tensorflow::Status::OK();
}

tensorflow::Status LoadDynamicModelConfig(
    const ::google::protobuf::Any& any,
    Target<std::unique_ptr<Loader>>* target) {
  CHECK(false)  // Crash ok
      << "ModelServer does not yet support dynamic model config.";
}

ModelServerConfig BuildSingleModelConfig(const string& model_name,
                                         const string& model_base_path) {
  ModelServerConfig config;
  LOG(INFO) << "Building single TensorFlow model file config: "
            << " model_name: " << model_name
            << " model_base_path: " << model_base_path;
  tensorflow::serving::ModelConfig* single_model =
      config.mutable_model_config_list()->add_config();
  single_model->set_name(model_name);
  single_model->set_base_path(model_base_path);
  single_model->set_model_type(kTensorFlowModelType);
  return config;
}

grpc::Status ToGRPCStatus(const tensorflow::Status& status) {
  const int kErrorMessageLimit = 1024;
  string error_message;
  if (status.error_message().length() > kErrorMessageLimit) {
    LOG(ERROR) << "Truncating error: " << status.error_message();
    error_message =
        status.error_message().substr(0, kErrorMessageLimit) + "...TRUNCATED";
  } else {
    error_message = status.error_message();
  }
  return grpc::Status(static_cast<grpc::StatusCode>(status.code()),
                      error_message);
}

class PredictionServiceImpl final : public PredictionService::Service {
 public:
  explicit PredictionServiceImpl(std::unique_ptr<ServerCore> core)
      : core_(std::move(core)) {}

  grpc::Status Predict(ServerContext* context, const PredictRequest* request,
                       PredictResponse* response) override {
    return ToGRPCStatus(
        TensorflowPredictImpl::Predict(core_.get(), *request, response));
  }

 private:
  std::unique_ptr<ServerCore> core_;
};

void RunServer(int port, std::unique_ptr<ServerCore> core) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);
  PredictionServiceImpl service(std::move(core));
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running ModelServer at " << server_address << " ...";
  server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::int32 port = 8500;
  bool enable_batching = false;
  tensorflow::string model_name = "default";
  tensorflow::string model_base_path;
  const bool parse_result = tensorflow::ParseFlags(
      &argc, argv, {tensorflow::Flag("port", &port),
                    tensorflow::Flag("enable_batching", &enable_batching),
                    tensorflow::Flag("model_name", &model_name),
                    tensorflow::Flag("model_base_path", &model_base_path)});
  if (!parse_result || model_base_path.empty()) {
    std::cout << "Usage: model_server"
              << " [--port=8500]"
              << " [--enable_batching]"
              << " [--model_name=my_name]"
              << " --model_base_path=/path/to/export" << std::endl;
    return -1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  ModelServerConfig config =
      BuildSingleModelConfig(model_name, model_base_path);

  SessionBundleSourceAdapterConfig source_adapter_config;
  // Batching config
  if (enable_batching) {
    BatchingParameters* batching_parameters =
        source_adapter_config.mutable_config()->mutable_batching_parameters();
    batching_parameters->mutable_thread_pool_name()->set_value(
        "model_server_batch_threads");
  }

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(
      config, std::bind(CreateSourceAdapter, source_adapter_config,
                        std::placeholders::_1, std::placeholders::_2),
      &CreateServableStateMonitor, &LoadDynamicModelConfig, &core));
  RunServer(port, std::move(core));

  return 0;
}
