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

#include "tensorflow_serving/model_servers/server.h"

#include <unistd.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>

#include "google/protobuf/wrappers.pb.h"
#include "grpc/grpc.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/monitoring_config.pb.h"
#include "tensorflow_serving/config/ssl_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace main {

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

// Parses an ascii PlatformConfigMap protobuf from 'file'.
tensorflow::serving::PlatformConfigMap ParsePlatformConfigMap(
    const string& file) {
  tensorflow::serving::PlatformConfigMap platform_config_map;
  TF_CHECK_OK(ParseProtoTextFile(file, &platform_config_map));
  return platform_config_map;
}

// Parses an ascii SSLConfig protobuf from 'file'.
SSLConfig ParseSSLConfig(const string& file) {
  SSLConfig ssl_config;
  TF_CHECK_OK(ParseProtoTextFile(file, &ssl_config));
  return ssl_config;
}

// If 'ssl_config_file' is non-empty, build secure server credentials otherwise
// insecure channel
std::shared_ptr<::grpc::ServerCredentials>
BuildServerCredentialsFromSSLConfigFile(const string& ssl_config_file) {
  if (ssl_config_file.empty()) {
    return ::grpc::InsecureServerCredentials();
  }

  SSLConfig ssl_config = ParseSSLConfig(ssl_config_file);

  ::grpc::SslServerCredentialsOptions ssl_ops(
      ssl_config.client_verify()
          ? GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY
          : GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE);

  ssl_ops.force_client_auth = ssl_config.client_verify();

  if (ssl_config.custom_ca().size() > 0) {
    ssl_ops.pem_root_certs = ssl_config.custom_ca();
  }

  ::grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {
      ssl_config.server_key(), ssl_config.server_cert()};

  ssl_ops.pem_key_cert_pairs.push_back(keycert);

  return ::grpc::SslServerCredentials(ssl_ops);
}

}  // namespace

Server::Options::Options()
    : model_name("default"),
      saved_model_tags(tensorflow::kSavedModelTagServe) {}

Server::~Server() { WaitForTermination(); }

std::atomic<bool> Server::model_conf_mon_thread_exit_(false);

int Server::model_config_mon_thread_function(ServerCore* core_p, const string& model_conf_file,
                                             tensorflow::int32 modelconf_poll_wait_seconds) {

  LOG(INFO) << "Enter model_config_mon_thread_function";

  std::string prev_conf_msg = "";

  while(1) {

    int time_left_sec = modelconf_poll_wait_seconds;

    while(time_left_sec > 0) {
      tensorflow::Env::Default()->SleepForMicroseconds(100 * 10000 * 1 /* 1 seconds */);
      time_left_sec -= 1;

      if (model_conf_mon_thread_exit_.load()) {
        break;
      }
    }

    if (model_conf_mon_thread_exit_.load()) {
      break;
    }

    LOG(INFO) << "Looking at modelconf from " << model_conf_file;

    ModelServerConfig new_config = ReadProtoFromFile<ModelServerConfig>(model_conf_file);

    std::string new_conf_msg;
    new_config.SerializeToString(&new_conf_msg);

    if (new_conf_msg != prev_conf_msg) {
      LOG(INFO) << "Reloading modelconf";

      prev_conf_msg = new_conf_msg;

      Status status = core_p->ReloadConfig(new_config);

      if (!status.ok()) {
        LOG(ERROR) << "Reload modelconf failed: " << status.error_message();
      } else {
        LOG(INFO) << "Running modelconf success";
      }
    }
  }

  LOG(INFO) << "Exiting model_config_mon_thread_function";

  return 0;
}

Status Server::BuildAndStart(const Options& server_options) {
  const bool use_saved_model = true;

  if (server_options.grpc_port == 0) {
    return errors::InvalidArgument("server_options.grpc_port is not set.");
  }

  if (server_options.model_base_path.empty() &&
      server_options.model_config_file.empty()) {
    return errors::InvalidArgument(
        "Both server_options.model_base_path and "
        "server_options.model_config_file are empty!");
  }

  // For ServerCore Options, we leave servable_state_monitor_creator unspecified
  // so the default servable_state_monitor_creator will be used.
  ServerCore::Options options;

  // model server config
  if (server_options.model_config_file.empty()) {
    options.model_server_config = BuildSingleModelConfig(
        server_options.model_name, server_options.model_base_path);
  } else {
    options.model_server_config =
        ReadProtoFromFile<ModelServerConfig>(server_options.model_config_file);
  }

  if (server_options.platform_config_file.empty()) {
    SessionBundleConfig session_bundle_config;
    // Batching config
    if (server_options.enable_batching) {
      BatchingParameters* batching_parameters =
          session_bundle_config.mutable_batching_parameters();
      if (server_options.batching_parameters_file.empty()) {
        batching_parameters->mutable_thread_pool_name()->set_value(
            "model_server_batch_threads");
      } else {
        *batching_parameters = ReadProtoFromFile<BatchingParameters>(
            server_options.batching_parameters_file);
      }
    } else if (!server_options.batching_parameters_file.empty()) {
      return errors::InvalidArgument(
          "server_options.batching_parameters_file is set without setting "
          "server_options.enable_batching to true.");
    }

    session_bundle_config.mutable_session_config()
        ->mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(
            server_options.per_process_gpu_memory_fraction);
    session_bundle_config.mutable_session_config()
        ->set_intra_op_parallelism_threads(
            server_options.tensorflow_session_parallelism);
    session_bundle_config.mutable_session_config()
        ->set_inter_op_parallelism_threads(
            server_options.tensorflow_session_parallelism);
    const std::vector<string> tags =
        tensorflow::str_util::Split(server_options.saved_model_tags, ",");
    for (const string& tag : tags) {
      *session_bundle_config.add_saved_model_tags() = tag;
    }
    session_bundle_config.set_enable_model_warmup(
        server_options.enable_model_warmup);
    options.platform_config_map = CreateTensorFlowPlatformConfigMap(
        session_bundle_config, use_saved_model);
  } else {
    options.platform_config_map =
        ParsePlatformConfigMap(server_options.platform_config_file);
  }

  options.custom_model_config_loader = &LoadCustomModelConfig;
  options.aspired_version_policy =
      std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
  options.max_num_load_retries = server_options.max_num_load_retries;
  options.load_retry_interval_micros =
      server_options.load_retry_interval_micros;
  options.file_system_poll_wait_seconds =
      server_options.file_system_poll_wait_seconds;
  options.flush_filesystem_caches = server_options.flush_filesystem_caches;

  TF_RETURN_IF_ERROR(ServerCore::Create(std::move(options), &server_core_));

  if (server_options.modelconf_poll_wait_seconds > 0) {
    LOG(INFO) << "Starting modelconf monitor ";
    model_conf_mon_thread_ = std::unique_ptr<std::thread>(new std::thread(&Server::model_config_mon_thread_function,
                                                                          server_core_.get(),
                                                                          server_options.model_config_file,
                                                                          server_options.modelconf_poll_wait_seconds));
  }

  // 0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address =
      "0.0.0.0:" + std::to_string(server_options.grpc_port);
  model_service_ = absl::make_unique<ModelServiceImpl>(server_core_.get());
  prediction_service_ = absl::make_unique<PredictionServiceImpl>(
      server_core_.get(), use_saved_model);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(
      server_address,
      BuildServerCredentialsFromSSLConfigFile(server_options.ssl_config_file));
  builder.RegisterService(model_service_.get());
  builder.RegisterService(prediction_service_.get());
  builder.SetMaxMessageSize(tensorflow::kint32max);
  const std::vector<GrpcChannelArgument> channel_arguments =
      parseGrpcChannelArgs(server_options.grpc_channel_arguments);
  for (GrpcChannelArgument channel_argument : channel_arguments) {
    // gRPC accept arguments of two types, int and string. We will attempt to
    // parse each arg as int and pass it on as such if successful. Otherwise we
    // will pass it as a string. gRPC will log arguments that were not accepted.
    tensorflow::int32 value;
    if (tensorflow::strings::safe_strto32(channel_argument.value, &value)) {
      builder.AddChannelArgument(channel_argument.key, value);
    } else {
      builder.AddChannelArgument(channel_argument.key, channel_argument.value);
    }
  }
  grpc_server_ = builder.BuildAndStart();
  if (grpc_server_ == nullptr) {
    return errors::InvalidArgument("Failed to BuildAndStart gRPC server");
  }
  LOG(INFO) << "Running gRPC ModelServer at " << server_address << " ...";

  if (server_options.http_port != 0) {
    if (server_options.http_port != server_options.grpc_port) {
      const string server_address =
          "localhost:" + std::to_string(server_options.http_port);
      MonitoringConfig monitoring_config;
      if (!server_options.monitoring_config_file.empty()) {
        monitoring_config = ReadProtoFromFile<MonitoringConfig>(
            server_options.monitoring_config_file);
      }
      http_server_ = CreateAndStartHttpServer(
          server_options.http_port, server_options.http_num_threads,
          server_options.http_timeout_in_ms, monitoring_config,
          server_core_.get());
      if (http_server_ != nullptr) {
        LOG(INFO) << "Exporting HTTP/REST API at:" << server_address << " ...";
      } else {
        LOG(ERROR) << "Failed to start HTTP Server at " << server_address;
      }
    } else {
      LOG(ERROR) << "server_options.http_port cannot be same as grpc_port. "
                 << "Please use a different port for HTTP/REST API. "
                 << "Skipped exporting HTTP/REST API.";
    }
  }
  return Status::OK();
}

void Server::WaitForTermination() {
  if (http_server_ != nullptr) {
    http_server_->WaitForTermination();
  }
  if (grpc_server_ != nullptr) {
    grpc_server_->Wait();
  }
  if (model_conf_mon_thread_ != nullptr) {
      model_conf_mon_thread_exit_.store(true);
      model_conf_mon_thread_->join();
  }
}

}  // namespace main
}  // namespace serving
}  // namespace tensorflow
