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
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/logging.pb.h"

namespace tensorflow {
namespace serving {

// static
Status ServerRequestLogger::Create(
    const std::function<Status(const LoggingConfig& logging_config,
                               std::unique_ptr<RequestLogger>*)>&
        request_logger_creator,
    std::unique_ptr<ServerRequestLogger>* server_request_logger) {
  server_request_logger->reset(new ServerRequestLogger(request_logger_creator));
  return Status::OK();
}

ServerRequestLogger::ServerRequestLogger(
    const std::function<Status(const LoggingConfig& logging_config,
                               std::unique_ptr<RequestLogger>*)>&
        request_logger_creator)
    : request_logger_map_(
          std::unique_ptr<RequestLoggerMap>(new RequestLoggerMap())),
      request_logger_creator_(request_logger_creator) {}

Status ServerRequestLogger::Update(
    const std::map<string, LoggingConfig>& logging_config_map) {
  if (!logging_config_map.empty() && !request_logger_creator_) {
    return errors::InvalidArgument("No request-logger-creator provided.");
  }
  std::set<string> filename_prefixes;
  std::unique_ptr<RequestLoggerMap> request_logger_map(new RequestLoggerMap());
  for (const auto& model_and_logging_config : logging_config_map) {
    auto& request_logger =
        (*request_logger_map)[model_and_logging_config.first];
    const string& filename_prefix =
        model_and_logging_config.second.log_collector_config()
            .filename_prefix();
    if (!gtl::InsertIfNotPresent(&filename_prefixes, filename_prefix)) {
      // Logs for each model is supposed to be separated from each other.
      return errors::InvalidArgument(
          "Duplicate LogCollectorConfig::filename_prefix(): ", filename_prefix);
    }
    TF_RETURN_IF_ERROR(request_logger_creator_(model_and_logging_config.second,
                                               &request_logger));
  }
  request_logger_map_.Update(std::move(request_logger_map));
  return Status::OK();
}

Status ServerRequestLogger::Log(const google::protobuf::Message& request,
                                const google::protobuf::Message& response,
                                const LogMetadata& log_metadata) {
  const string& model_name = log_metadata.model_spec().name();
  auto request_logger_map = request_logger_map_.get();
  if (request_logger_map->empty()) {
    VLOG(2) << "Request logger map is empty.";
    return Status::OK();
  }
  auto found_it = request_logger_map->find(model_name);
  if (found_it == request_logger_map->end()) {
    VLOG(2) << "Cannot find request-logger for model: " << model_name;
    return Status::OK();
  }
  auto& request_logger = found_it->second;
  return request_logger->Log(request, response, log_metadata);
}

}  // namespace serving
}  // namespace tensorflow
