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

#include "tensorflow_serving/core/server_request_logger.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/logging.pb.h"

namespace tensorflow {
namespace serving {

// static
Status ServerRequestLogger::Create(
    const std::map<string, LoggingConfig>& logging_config_map,
    const std::function<Status(const LogCollectorConfig& log_collector_config,
                               std::unique_ptr<RequestLogger>*)>&
        request_logger_creator,
    std::unique_ptr<ServerRequestLogger>* server_request_logger) {
  std::unordered_map<string, std::unique_ptr<RequestLogger>> request_logger_map;
  for (const auto& model_and_logging_config : logging_config_map) {
    auto& request_logger = request_logger_map[model_and_logging_config.first];
    TF_RETURN_IF_ERROR(request_logger_creator(
        model_and_logging_config.second.log_collector_config(),
        &request_logger));
  }
  server_request_logger->reset(
      new ServerRequestLogger(std::move(request_logger_map)));
  return Status::OK();
}

ServerRequestLogger::ServerRequestLogger(
    std::unordered_map<string, std::unique_ptr<RequestLogger>>
        request_logger_map)
    : request_logger_map_(std::move(request_logger_map)) {}

Status ServerRequestLogger::Log(const google::protobuf::Message& request,
                                const google::protobuf::Message& response,
                                const LogMetadata& log_metadata) {
  const string& model_name = log_metadata.model_spec().name();
  auto found_it = request_logger_map_.find(model_name);
  if (found_it == request_logger_map_.end()) {
    const string error =
        strings::StrCat("Cannot find request-logger for model: ", model_name);
    // This shouldn't happen at all, so dchecking for capturing in tests.
    DCHECK(false) << error;  // Crash ok.
    return errors::NotFound(error);
  }
  auto& request_logger = found_it->second;
  return request_logger->Log(request, response, log_metadata);
}

}  // namespace serving
}  // namespace tensorflow
