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
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/apis/model.pb.h"

namespace tensorflow {
namespace serving {

// static
Status ServerRequestLogger::Create(
    LoggerCreator request_logger_creator,
    std::unique_ptr<ServerRequestLogger>* server_request_logger) {
  server_request_logger->reset(
      new ServerRequestLogger(std::move(request_logger_creator)));
  return OkStatus();
}

ServerRequestLogger::ServerRequestLogger(LoggerCreator request_logger_creator)
    : request_logger_creator_(std::move(request_logger_creator)) {}

Status ServerRequestLogger::FindOrCreateLogger(
    const LoggingConfig& config,
    StringToUniqueRequestLoggerMap* new_config_to_logger_map,
    RequestLogger** result) {
  string serialized_config;
  if (!SerializeToStringDeterministic(config, &serialized_config)) {
    return errors::InvalidArgument("Cannot serialize config.");
  }

  auto find_new_it = new_config_to_logger_map->find(serialized_config);
  if (find_new_it != new_config_to_logger_map->end()) {
    // The logger is already in new_config_to_logger_map, simply return it.
    *result = find_new_it->second.get();
    return OkStatus();
  }

  auto find_old_it = config_to_logger_map_.find(serialized_config);
  if (find_old_it != config_to_logger_map_.end()) {
    // The logger is in old_config_to_logger_map. Move it to
    // new_config_to_logger_map, erase the entry in config_to_logger_map_ and
    // return the logger.
    *result = find_old_it->second.get();
    new_config_to_logger_map->emplace(
        std::make_pair(serialized_config, std::move(find_old_it->second)));
    config_to_logger_map_.erase(find_old_it);
    return OkStatus();
  }

  // The logger does not exist. Create a new logger, insert it into
  // new_config_to_logger_map and return it.
  std::unique_ptr<RequestLogger> logger;
  TF_RETURN_IF_ERROR(request_logger_creator_(config, &logger));
  *result = logger.get();
  new_config_to_logger_map->emplace(
      std::make_pair(serialized_config, std::move(logger)));
  return OkStatus();
}

Status ServerRequestLogger::Update(
    const std::map<string, std::vector<LoggingConfig>>& logging_config_map) {
  if (!logging_config_map.empty() && !request_logger_creator_) {
    return errors::InvalidArgument("No request-logger-creator provided.");
  }

  // Those new maps will only contain loggers from logging_config_map and
  // replace the current versions further down.
  std::unique_ptr<StringToRequestLoggersMap> new_model_to_loggers_map(
      new StringToRequestLoggersMap());
  StringToUniqueRequestLoggerMap new_config_to_logger_map;

  mutex_lock l(update_mu_);

  for (const auto& model_and_logging_config : logging_config_map) {
    for (const auto& logging_config : model_and_logging_config.second) {
      RequestLogger* logger;
      TF_RETURN_IF_ERROR(FindOrCreateLogger(
          logging_config, &new_config_to_logger_map, &logger));
      const string& model_name = model_and_logging_config.first;
      (*new_model_to_loggers_map)[model_name].push_back(logger);
    }
  }

  model_to_loggers_map_.Update(std::move(new_model_to_loggers_map));
  // Any remaining loggers in config_to_logger_map_ will not be needed anymore
  // and destructed at this point.
  config_to_logger_map_ = std::move(new_config_to_logger_map);

  return OkStatus();
}

Status ServerRequestLogger::Log(const google::protobuf::Message& request,
                                const google::protobuf::Message& response,
                                const LogMetadata& log_metadata) {
  const string& model_name = log_metadata.model_spec().name();
  auto model_to_loggers_map = model_to_loggers_map_.get();
  if (!model_to_loggers_map || model_to_loggers_map->empty()) {
    VLOG(2) << "Request loggers map is empty.";
    return OkStatus();
  }
  auto found_it = model_to_loggers_map->find(model_name);
  if (found_it == model_to_loggers_map->end()) {
    VLOG(2) << "Cannot find request-loggers for model: " << model_name;
    return OkStatus();
  }

  Status status;
  for (const auto& logger : found_it->second) {
    // Note: Only first error will be tracked/returned.
    status.Update(logger->Log(request, response, log_metadata));
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
