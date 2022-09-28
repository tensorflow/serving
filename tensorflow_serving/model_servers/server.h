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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_H_

#include <memory>

#include "grpcpp/server.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow_serving/model_servers/http_server.h"
#include "tensorflow_serving/model_servers/model_service_impl.h"
#include "tensorflow_serving/model_servers/prediction_service_impl.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {
namespace main {

class Server {
 public:
  struct Options {
    //
    // gRPC Server options
    //
    tensorflow::int32 grpc_port = 8500;
    tensorflow::string grpc_channel_arguments;
    tensorflow::string grpc_socket_path;
    tensorflow::int32 grpc_max_threads = 4.0 * port::NumSchedulableCPUs();

    //
    // HTTP Server options.
    //
    tensorflow::int32 http_port = 0;
    tensorflow::int32 http_num_threads = 4.0 * port::NumSchedulableCPUs();
    tensorflow::int32 http_timeout_in_ms = 30000;  // 30 seconds.
    bool enable_cors_support = false;

    //
    // Model Server options.
    //
    bool enable_batching = false;
    bool enable_per_model_batching_params = false;
    bool allow_version_labels_for_unavailable_models = false;
    bool force_allow_any_version_labels_for_unavailable_models = false;
    float per_process_gpu_memory_fraction = 0;
    tensorflow::string batching_parameters_file;
    tensorflow::string model_name;
    tensorflow::int32 num_load_threads = 0;
    tensorflow::int32 num_unload_threads = 0;
    tensorflow::int32 max_num_load_retries = 5;
    int64_t load_retry_interval_micros = 1LL * 60 * 1000 * 1000;
    tensorflow::int32 file_system_poll_wait_seconds = 1;
    bool flush_filesystem_caches = true;
    tensorflow::string model_base_path;
    tensorflow::string saved_model_tags;
    // Tensorflow session parallelism of zero means that both inter and intra op
    // thread pools will be auto configured.
    int64_t tensorflow_session_parallelism = 0;

    // Zero means that the thread pools will be auto configured.
    int64_t tensorflow_intra_op_parallelism = 0;
    int64_t tensorflow_inter_op_parallelism = 0;
    tensorflow::string platform_config_file;
    // Only one of ALTS or SSl can be specified. I.e. either
    // use_alts_credentials must be false or ssl_config_file must be empty.
    bool use_alts_credentials = false;
    tensorflow::string ssl_config_file;
    string model_config_file;
    // Text proto file for TensorFlow Session ConfigProto.
    string tensorflow_session_config_file;
    // Zero means server will not poll FS for model config file after start-up.
    tensorflow::int32 fs_model_config_poll_wait_seconds = 0;
    bool enable_model_warmup = true;
    // This value is used only if > 0.
    tensorflow::int32 num_request_iterations_for_warmup = 0;
    tensorflow::string monitoring_config_file;
    // Tensorflow session run options.
    bool enforce_session_run_timeout = true;
    bool remove_unused_fields_from_bundle_metagraph = true;
    bool prefer_tflite_model = false;
    tensorflow::int32 num_tflite_pools = port::NumSchedulableCPUs();
    tensorflow::int32 num_tflite_interpreters_per_pool = 1;
    tensorflow::string thread_pool_factory_config_file;
    bool enable_signature_method_name_check = false;
    bool enable_profiler = true;

    Options();
  };

  // Blocks the current thread waiting for servers (if any)
  // started as part of BuildAndStart() call.
  ~Server();

  // Build and start gRPC (and optionally HTTP) server, to be ready to
  // accept and process new requests over gRPC (and optionally HTTP/REST).
  Status BuildAndStart(const Options& server_options);

  // Wait for servers started in BuildAndStart() above to terminate.
  // This will block the current thread until termination is successful.
  void WaitForTermination();

 private:
  // Polls the filesystem, parses config at specified path, and calls
  // ServerCore::ReloadConfig with the captured model config.
  void PollFilesystemAndReloadConfig(const string& config_file_path);

  std::unique_ptr<ServerCore> server_core_;
  std::unique_ptr<ModelServiceImpl> model_service_;
  std::unique_ptr<PredictionService::Service> prediction_service_;
  std::unique_ptr<tensorflow::grpc::ProfilerService::Service> profiler_service_;
  std::unique_ptr<::grpc::Server> grpc_server_;
  std::unique_ptr<net_http::HTTPServerInterface> http_server_;
  // A thread that calls PollFilesystemAndReloadConfig() periodically if
  // fs_model_config_poll_wait_seconds > 0.
  std::unique_ptr<PeriodicFunction> fs_config_polling_thread_;
  std::unique_ptr<ThreadPoolFactory> thread_pool_factory_;
};

}  // namespace main
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_SERVER_H_
