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
// models will be loaded and managed over time using the
// AvailabilityPreservingPolicy at:
//     tensorflow_serving/core/availability_preserving_policy.h.
// by AspiredVersionsManager at:
//     tensorflow_serving/core/aspired_versions_manager.h
//
// ModelServer has inter-request batching support built-in, by using the
// BatchingSession at:
//     tensorflow_serving/batching/batching_session.h
//
// To serve a single model, run with:
//     $path_to_binary/tensorflow_model_server \
//     --model_base_path=[/tmp/my_model | gs://gcs_address]
// IMPORTANT: Be sure the base path excludes the version directory. For
// example for a model at /tmp/my_model/123, where 123 is the version, the base
// path is /tmp/my_model.
//
// To specify model name (default "default"): --model_name=my_name
// To specify port (default 8500): --port=my_port
// To enable batching (default disabled): --enable_batching
// To override the default batching parameters: --batching_parameters_file

#include <unistd.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc/grpc.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/apis/prediction_service.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/model_service_impl.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace grpc {
class ServerCompletionQueue;
}  // namespace grpc

using tensorflow::string;
using tensorflow::serving::AspiredVersionPolicy;
using tensorflow::serving::AspiredVersionsManager;
using tensorflow::serving::AvailabilityPreservingPolicy;
using tensorflow::serving::BatchingParameters;
using tensorflow::serving::EventBus;
using tensorflow::serving::FileSystemStoragePathSourceConfig;
using tensorflow::serving::GetModelMetadataImpl;
using tensorflow::serving::ModelServerConfig;
using tensorflow::serving::ServableState;
using tensorflow::serving::ServerCore;
using tensorflow::serving::SessionBundleConfig;
using tensorflow::serving::TensorflowClassificationServiceImpl;
using tensorflow::serving::TensorflowRegressionServiceImpl;
using tensorflow::serving::TensorflowPredictor;
using tensorflow::serving::TensorflowRegressionServiceImpl;
using tensorflow::serving::UniquePtrWithDeps;

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using tensorflow::serving::ClassificationRequest;
using tensorflow::serving::ClassificationResponse;
using tensorflow::serving::GetModelMetadataRequest;
using tensorflow::serving::GetModelMetadataResponse;
using tensorflow::serving::MultiInferenceRequest;
using tensorflow::serving::MultiInferenceResponse;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::RegressionRequest;
using tensorflow::serving::RegressionResponse;
using tensorflow::serving::PredictionService;

namespace {

tensorflow::Status ParseProtoTextFile(const string& file,
                                      google::protobuf::Message* message) {
  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file_data;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewReadOnlyMemoryRegionFromFile(file,
                                                                  &file_data));
  string file_data_str(static_cast<const char*>(file_data->data()),
                       file_data->length());
  if (tensorflow::protobuf::TextFormat::ParseFromString(file_data_str,
                                                        message)) {
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::InvalidArgument("Invalid protobuf file: '", file,
                                               "'");
  }
}

tensorflow::Status LoadCustomModelConfig(
    const ::google::protobuf::Any& any,
    EventBus<ServableState>* servable_event_bus,
    UniquePtrWithDeps<AspiredVersionsManager>* manager) {
  LOG(FATAL)  // Crash ok
      << "ModelServer does not yet support custom model config.";
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
  single_model->set_model_platform(
      tensorflow::serving::kTensorFlowModelPlatform);
  return config;
}

template <typename ProtoType>
ProtoType ReadProtoFromFile(const string& file) {
  ProtoType proto;
  TF_CHECK_OK(ParseProtoTextFile(file, &proto));
  return proto;
}

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

class PredictionServiceImpl final : public PredictionService::Service {
 public:
  explicit PredictionServiceImpl(ServerCore* core, bool use_saved_model)
      : core_(core),
        predictor_(new TensorflowPredictor(use_saved_model)),
        use_saved_model_(use_saved_model) {}

  grpc::Status Predict(ServerContext* context, const PredictRequest* request,
                       PredictResponse* response) override {
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    // By default, this is infinite which is the same default as RunOptions.
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
    const grpc::Status status = tensorflow::serving::ToGRPCStatus(
        predictor_->Predict(run_options, core_, *request, response));
    if (!status.ok()) {
      VLOG(1) << "Predict failed: " << status.error_message();
    }
    return status;
  }

  grpc::Status GetModelMetadata(ServerContext* context,
                                const GetModelMetadataRequest* request,
                                GetModelMetadataResponse* response) override {
    if (!use_saved_model_) {
      return tensorflow::serving::ToGRPCStatus(
          tensorflow::errors::InvalidArgument(
              "GetModelMetadata API is only available when use_saved_model is "
              "set to true"));
    }
    const grpc::Status status = tensorflow::serving::ToGRPCStatus(
        GetModelMetadataImpl::GetModelMetadata(core_, *request, response));
    if (!status.ok()) {
      VLOG(1) << "GetModelMetadata failed: " << status.error_message();
    }
    return status;
  }

  grpc::Status Classify(ServerContext* context,
                        const ClassificationRequest* request,
                        ClassificationResponse* response) override {
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    // By default, this is infinite which is the same default as RunOptions.
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
    const grpc::Status status = tensorflow::serving::ToGRPCStatus(
        TensorflowClassificationServiceImpl::Classify(run_options, core_,
                                                      *request, response));
    if (!status.ok()) {
      VLOG(1) << "Classify request failed: " << status.error_message();
    }
    return status;
  }

  grpc::Status Regress(ServerContext* context, const RegressionRequest* request,
                       RegressionResponse* response) override {
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    // By default, this is infinite which is the same default as RunOptions.
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
    const grpc::Status status = tensorflow::serving::ToGRPCStatus(
        TensorflowRegressionServiceImpl::Regress(run_options, core_, *request,
                                                 response));
    if (!status.ok()) {
      VLOG(1) << "Regress request failed: " << status.error_message();
    }
    return status;
  }

  grpc::Status MultiInference(ServerContext* context,
                              const MultiInferenceRequest* request,
                              MultiInferenceResponse* response) override {
    tensorflow::RunOptions run_options = tensorflow::RunOptions();
    // By default, this is infinite which is the same default as RunOptions.
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
    const grpc::Status status = tensorflow::serving::ToGRPCStatus(
        RunMultiInference(run_options, core_, *request, response));
    if (!status.ok()) {
      VLOG(1) << "MultiInference request failed: " << status.error_message();
    }
    return status;
  }

 private:
  ServerCore* core_;
  std::unique_ptr<TensorflowPredictor> predictor_;
  bool use_saved_model_;
};

// gRPC Channel Arguments to be passed from command line to gRPC ServerBuilder.
struct GrpcChannelArgument {
  string key;
  string value;
};

// Parses a comma separated list of gRPC channel arguments into list of
// ChannelArgument.
std::vector<GrpcChannelArgument> parseGrpcChannelArgs(
    const string& channel_arguments_str) {
  const std::vector<string> channel_arguments =
      tensorflow::str_util::Split(channel_arguments_str, ",");
  std::vector<GrpcChannelArgument> result;
  for (const string& channel_argument : channel_arguments) {
    const std::vector<string> key_val =
        tensorflow::str_util::Split(channel_argument, "=");
    result.push_back({key_val[0], key_val[1]});
  }
  return result;
}

void RunServer(int port, std::unique_ptr<ServerCore> core, bool use_saved_model,
               const string& grpc_channel_arguments) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);
  tensorflow::serving::ModelServiceImpl model_service(core.get());
  PredictionServiceImpl prediction_service(core.get(), use_saved_model);
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&model_service);
  builder.RegisterService(&prediction_service);
  builder.SetMaxMessageSize(tensorflow::kint32max);
  const std::vector<GrpcChannelArgument> channel_arguments =
      parseGrpcChannelArgs(grpc_channel_arguments);
  for (GrpcChannelArgument channel_argument : channel_arguments) {
    // gRPC accept arguments of two types, int and string. We will attempt to
    // parse each arg as int and pass it on as such if successful. Otherwise we
    // will pass it as a string. gRPC will log arguments that were not accepted.
    int value;
    if (tensorflow::strings::safe_strto32(channel_argument.key, &value)) {
      builder.AddChannelArgument(channel_argument.key, value);
    } else {
      builder.AddChannelArgument(channel_argument.key, channel_argument.value);
    }
  }
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running ModelServer at " << server_address << " ...";
  server->Wait();
}

// Parses an ascii PlatformConfigMap protobuf from 'file'.
tensorflow::serving::PlatformConfigMap ParsePlatformConfigMap(
    const string& file) {
  tensorflow::serving::PlatformConfigMap platform_config_map;
  TF_CHECK_OK(ParseProtoTextFile(file, &platform_config_map));
  return platform_config_map;
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::int32 port = 8500;
  bool enable_batching = false;
  float per_process_gpu_memory_fraction = 0;
  tensorflow::string batching_parameters_file;
  tensorflow::string model_name = "default";
  tensorflow::int32 file_system_poll_wait_seconds = 1;
  bool flush_filesystem_caches = true;
  tensorflow::string model_base_path;
  const bool use_saved_model = true;
  tensorflow::string saved_model_tags = tensorflow::kSavedModelTagServe;
  // Tensorflow session parallelism of zero means that both inter and intra op
  // thread pools will be auto configured.
  tensorflow::int64 tensorflow_session_parallelism = 0;
  string platform_config_file = "";
  string model_config_file;
  string grpc_channel_arguments = "";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("port", &port, "port to listen on"),
      tensorflow::Flag("enable_batching", &enable_batching, "enable batching"),
      tensorflow::Flag("batching_parameters_file", &batching_parameters_file,
                       "If non-empty, read an ascii BatchingParameters "
                       "protobuf from the supplied file name and use the "
                       "contained values instead of the defaults."),
      tensorflow::Flag("model_config_file", &model_config_file,
                       "If non-empty, read an ascii ModelServerConfig "
                       "protobuf from the supplied file name, and serve the "
                       "models in that file. This config file can be used to "
                       "specify multiple models to serve and other advanced "
                       "parameters including non-default version policy. (If "
                       "used, --model_name, --model_base_path are ignored.)"),
      tensorflow::Flag("model_name", &model_name,
                       "name of model (ignored "
                       "if --model_config_file flag is set"),
      tensorflow::Flag("model_base_path", &model_base_path,
                       "path to export (ignored if --model_config_file flag "
                       "is set, otherwise required)"),
      tensorflow::Flag("file_system_poll_wait_seconds",
                       &file_system_poll_wait_seconds,
                       "interval in seconds between each poll of the file "
                       "system for new model version"),
      tensorflow::Flag("flush_filesystem_caches", &flush_filesystem_caches,
                       "If true (the default), filesystem caches will be "
                       "flushed after the initial load of all servables, and "
                       "after each subsequent individual servable reload (if "
                       "the number of load threads is 1). This reduces memory "
                       "consumption of the model server, at the potential cost "
                       "of cache misses if model files are accessed after "
                       "servables are loaded."),
      tensorflow::Flag("tensorflow_session_parallelism",
                       &tensorflow_session_parallelism,
                       "Number of threads to use for running a "
                       "Tensorflow session. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      tensorflow::Flag("platform_config_file", &platform_config_file,
                       "If non-empty, read an ascii PlatformConfigMap protobuf "
                       "from the supplied file name, and use that platform "
                       "config instead of the Tensorflow platform. (If used, "
                       "--enable_batching is ignored.)"),
      tensorflow::Flag(
          "per_process_gpu_memory_fraction", &per_process_gpu_memory_fraction,
          "Fraction that each process occupies of the GPU memory space "
          "the value is between 0.0 and 1.0 (with 0.0 as the default) "
          "If 1.0, the server will allocate all the memory when the server "
          "starts, If 0.0, Tensorflow will automatically select a value."),
      tensorflow::Flag("saved_model_tags", &saved_model_tags,
                       "Comma-separated set of tags corresponding to the meta "
                       "graph def to load from SavedModel."),
      tensorflow::Flag("grpc_channel_arguments", &grpc_channel_arguments,
                       "A comma separated list of arguments to be passed to "
                       "the grpc server. (e.g. "
                       "grpc.max_connection_age_ms=2000)")};

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || (model_base_path.empty() && model_config_file.empty())) {
    std::cout << usage;
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 1) {
    std::cout << "unknown argument: " << argv[1] << "\n" << usage;
  }

  // For ServerCore Options, we leave servable_state_monitor_creator unspecified
  // so the default servable_state_monitor_creator will be used.
  ServerCore::Options options;

  // model server config
  if (model_config_file.empty()) {
    options.model_server_config =
        BuildSingleModelConfig(model_name, model_base_path);
  } else {
    options.model_server_config =
        ReadProtoFromFile<ModelServerConfig>(model_config_file);
  }

  if (platform_config_file.empty()) {
    SessionBundleConfig session_bundle_config;
    // Batching config
    if (enable_batching) {
      BatchingParameters* batching_parameters =
          session_bundle_config.mutable_batching_parameters();
      if (batching_parameters_file.empty()) {
        batching_parameters->mutable_thread_pool_name()->set_value(
            "model_server_batch_threads");
      } else {
        *batching_parameters =
            ReadProtoFromFile<BatchingParameters>(batching_parameters_file);
      }
    } else if (!batching_parameters_file.empty()) {
      LOG(FATAL)  // Crash ok
          << "You supplied --batching_parameters_file without "
             "--enable_batching";
    }

    session_bundle_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
    session_bundle_config.mutable_session_config()
        ->set_intra_op_parallelism_threads(tensorflow_session_parallelism);
    session_bundle_config.mutable_session_config()
        ->set_inter_op_parallelism_threads(tensorflow_session_parallelism);
    const std::vector<string> tags =
        tensorflow::str_util::Split(saved_model_tags, ",");
    for (const string& tag : tags) {
      *session_bundle_config.add_saved_model_tags() = tag;
    }
    options.platform_config_map = CreateTensorFlowPlatformConfigMap(
        session_bundle_config, use_saved_model);
  } else {
    options.platform_config_map = ParsePlatformConfigMap(platform_config_file);
  }

  options.custom_model_config_loader = &LoadCustomModelConfig;

  options.aspired_version_policy =
      std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
  options.file_system_poll_wait_seconds = file_system_poll_wait_seconds;
  options.flush_filesystem_caches = flush_filesystem_caches;

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(std::move(options), &core));
  RunServer(port, std::move(core), use_saved_model, grpc_channel_arguments);

  return 0;
}
