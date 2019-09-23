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

#include <iostream>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/model_servers/server.h"
#include "tensorflow_serving/model_servers/version.h"

int main(int argc, char** argv) {
  tensorflow::serving::main::Server::Options options;
  bool display_version = false;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("port", &options.grpc_port,
                       "Port to listen on for gRPC API"),
      tensorflow::Flag("grpc_socket_path", &options.grpc_socket_path,
                       "If non-empty, listen to a UNIX socket for gRPC API "
                       "on the given path. Can be either relative or absolute "
                       "path."),
      tensorflow::Flag("rest_api_port", &options.http_port,
                       "Port to listen on for HTTP/REST API. If set to zero "
                       "HTTP/REST API will not be exported. This port must be "
                       "different than the one specified in --port."),
      tensorflow::Flag("rest_api_num_threads", &options.http_num_threads,
                       "Number of threads for HTTP/REST API processing. If not "
                       "set, will be auto set based on number of CPUs."),
      tensorflow::Flag("rest_api_timeout_in_ms", &options.http_timeout_in_ms,
                       "Timeout for HTTP/REST API calls."),
      tensorflow::Flag("enable_batching", &options.enable_batching,
                       "enable batching"),
      tensorflow::Flag(
          "allow_version_labels_for_unavailable_models",
          &options.allow_version_labels_for_unavailable_models,
          "If true, allows assigning unused version labels to models that are "
          "not available yet."),
      tensorflow::Flag("batching_parameters_file",
                       &options.batching_parameters_file,
                       "If non-empty, read an ascii BatchingParameters "
                       "protobuf from the supplied file name and use the "
                       "contained values instead of the defaults."),
      tensorflow::Flag("model_config_file", &options.model_config_file,
                       "If non-empty, read an ascii ModelServerConfig "
                       "protobuf from the supplied file name, and serve the "
                       "models in that file. This config file can be used to "
                       "specify multiple models to serve and other advanced "
                       "parameters including non-default version policy. (If "
                       "used, --model_name, --model_base_path are ignored.)"),
      tensorflow::Flag("model_config_file_poll_wait_seconds",
                       &options.fs_model_config_poll_wait_seconds,
                       "Interval in seconds between each poll of the filesystem"
                       "for model_config_file. If unset or set to zero, "
                       "poll will be done exactly once and not periodically. "
                       "Setting this to negative is reserved for testing "
                       "purposes only."),
      tensorflow::Flag("model_name", &options.model_name,
                       "name of model (ignored "
                       "if --model_config_file flag is set)"),
      tensorflow::Flag("model_base_path", &options.model_base_path,
                       "path to export (ignored if --model_config_file flag "
                       "is set, otherwise required)"),
      tensorflow::Flag("max_num_load_retries", &options.max_num_load_retries,
                       "maximum number of times it retries loading a model "
                       "after the first failure, before giving up. "
                       "If set to 0, a load is attempted only once. "
                       "Default: 5"),
      tensorflow::Flag("load_retry_interval_micros",
                       &options.load_retry_interval_micros,
                       "The interval, in microseconds, between each servable "
                       "load retry. If set negative, it doesn't wait. "
                       "Default: 1 minute"),
      tensorflow::Flag("file_system_poll_wait_seconds",
                       &options.file_system_poll_wait_seconds,
                       "Interval in seconds between each poll of the "
                       "filesystem for new model version. If set to zero "
                       "poll will be exactly done once and not periodically. "
                       "Setting this to negative value will disable polling "
                       "entirely causing ModelServer to indefinitely wait for "
                       "a new model at startup. Negative values are reserved "
                       "for testing purposes only."),
      tensorflow::Flag("flush_filesystem_caches",
                       &options.flush_filesystem_caches,
                       "If true (the default), filesystem caches will be "
                       "flushed after the initial load of all servables, and "
                       "after each subsequent individual servable reload (if "
                       "the number of load threads is 1). This reduces memory "
                       "consumption of the model server, at the potential cost "
                       "of cache misses if model files are accessed after "
                       "servables are loaded."),
      tensorflow::Flag("tensorflow_session_parallelism",
                       &options.tensorflow_session_parallelism,
                       "Number of threads to use for running a "
                       "Tensorflow session. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      tensorflow::Flag("tensorflow_intra_op_parallelism",
                       &options.tensorflow_intra_op_parallelism,
                       "Number of threads to use to parallelize the execution"
                       "of an individual op. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      tensorflow::Flag("tensorflow_inter_op_parallelism",
                       &options.tensorflow_inter_op_parallelism,
                       "Controls the number of operators that can be executed "
                       "simultaneously. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      tensorflow::Flag(
          "ssl_config_file", &options.ssl_config_file,
          "If non-empty, read an ascii SSLConfig protobuf from "
          "the supplied file name and set up a secure gRPC channel"),
      tensorflow::Flag("platform_config_file", &options.platform_config_file,
                       "If non-empty, read an ascii PlatformConfigMap protobuf "
                       "from the supplied file name, and use that platform "
                       "config instead of the Tensorflow platform. (If used, "
                       "--enable_batching is ignored.)"),
      tensorflow::Flag(
          "per_process_gpu_memory_fraction",
          &options.per_process_gpu_memory_fraction,
          "Fraction that each process occupies of the GPU memory space "
          "the value is between 0.0 and 1.0 (with 0.0 as the default) "
          "If 1.0, the server will allocate all the memory when the server "
          "starts, If 0.0, Tensorflow will automatically select a value."),
      tensorflow::Flag("saved_model_tags", &options.saved_model_tags,
                       "Comma-separated set of tags corresponding to the meta "
                       "graph def to load from SavedModel."),
      tensorflow::Flag("grpc_channel_arguments",
                       &options.grpc_channel_arguments,
                       "A comma separated list of arguments to be passed to "
                       "the grpc server. (e.g. "
                       "grpc.max_connection_age_ms=2000)"),
      tensorflow::Flag("enable_model_warmup", &options.enable_model_warmup,
                       "Enables model warmup, which triggers lazy "
                       "initializations (such as TF optimizations) at load "
                       "time, to reduce first request latency."),
      tensorflow::Flag("version", &display_version, "Display version"),
      tensorflow::Flag(
          "monitoring_config_file", &options.monitoring_config_file,
          "If non-empty, read an ascii MonitoringConfig protobuf from "
          "the supplied file name"),
      tensorflow::Flag(
          "remove_unused_fields_from_bundle_metagraph",
          &options.remove_unused_fields_from_bundle_metagraph,
          "Removes unused fields from MetaGraphDef proto message to save "
          "memory."),
      tensorflow::Flag("use_tflite_model", &options.use_tflite_model,
                       "EXPERIMENTAL; CAN BE REMOVED ANYTIME! Load and use "
                       "TensorFlow Lite model from `model.tflite` file in "
                       "SavedModel directory instead of the TensorFlow model "
                       "from `saved_model.pb` file.")};

  const auto& usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    std::cout << usage;
    return -1;
  }

  if (display_version) {
    std::cout << "TensorFlow ModelServer: " << TF_Serving_Version() << "\n"
              << "TensorFlow Library: " << TF_Version() << "\n";
    return 0;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 1) {
    std::cout << "unknown argument: " << argv[1] << "\n" << usage;
  }

  tensorflow::serving::main::Server server;
  const auto& status = server.BuildAndStart(options);
  if (!status.ok()) {
    std::cout << "Failed to start server. Error: " << status << "\n";
    return -1;
  }
  server.WaitForTermination();
  return 0;
}
