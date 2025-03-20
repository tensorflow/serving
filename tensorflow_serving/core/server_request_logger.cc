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

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/request_logger.h"

namespace tensorflow {
namespace serving {

// static
absl::Status ServerRequestLogger::Create(
    LoggerCreator request_logger_creator,
    std::unique_ptr<ServerRequestLogger>* server_request_logger) {
  server_request_logger->reset(
      new ServerRequestLogger(std::move(request_logger_creator)));
  return absl::OkStatus();
}

ServerRequestLogger::ServerRequestLogger(LoggerCreator request_logger_creator)
    : request_logger_creator_(std::move(request_logger_creator)) {}

absl::Status ServerRequestLogger::FindOrCreateLogger(
    const LoggingConfig& config,
    StringToUniqueRequestLoggerMap* new_config_to_logger_map,
    std::shared_ptr<RequestLogger>* result) const {
  string serialized_config;
  if (!SerializeToStringDeterministic(config, &serialized_config)) {
    return errors::InvalidArgument("Cannot serialize config.");
  }

  auto find_new_it = new_config_to_logger_map->find(serialized_config);
  if (find_new_it != new_config_to_logger_map->end()) {
    // The logger is already in new_config_to_logger_map, simply return it.
    *result = find_new_it->second;
    return absl::OkStatus();
  }

  const auto find_old_it = config_to_logger_map_.find(serialized_config);
  if (find_old_it != config_to_logger_map_.end()) {
    // The logger is in old_config_to_logger_map. Create a copy to
    // new_config_to_logger_map. Note we cannot move, as entries in
    // config_to_logger_map_ should not be updated here.
    *result = find_old_it->second;
    new_config_to_logger_map->emplace(
        std::make_pair(serialized_config, find_old_it->second));
    return absl::OkStatus();
  }

  // The logger does not exist. Create a new logger, insert it into
  // new_config_to_logger_map and return it.
  TF_RETURN_IF_ERROR(request_logger_creator_(config, result));
  new_config_to_logger_map->emplace(std::make_pair(serialized_config, *result));
  return absl::OkStatus();
}

absl::Status ServerRequestLogger::Update(
    const std::map<string, std::vector<LoggingConfig>>& logging_config_map) {
  if (!logging_config_map.empty() && !request_logger_creator_) {
    return errors::InvalidArgument("No request-logger-creator provided.");
  }

  // Those new maps will only contain loggers from logging_config_map and
  // replace the current versions further down.
  std::unique_ptr<StringToRequestLoggersMap> new_model_to_loggers_map(
      new StringToRequestLoggersMap());
  StringToUniqueRequestLoggerMap new_config_to_logger_map;

  absl::MutexLock l(&update_mu_);

  for (const auto& model_and_logging_config : logging_config_map) {
    for (const auto& logging_config : model_and_logging_config.second) {
      std::shared_ptr<RequestLogger> logger;
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

  return absl::OkStatus();
}

absl::Status ServerRequestLogger::Log(const google::protobuf::Message& request,
                                      const google::protobuf::Message& response,
                                      const LogMetadata& log_metadata) {
  absl::Status status;
  InvokeLoggerForModel(
      log_metadata, [&status, &request, &response, &log_metadata](
                        const std::shared_ptr<RequestLogger>& logger) {
        // Note: Only first error will be tracked/returned.
        status.Update(logger->Log(request, response, log_metadata));
      });
  return status;
}

void ServerRequestLogger::InvokeLoggerForModel(
    const LogMetadata& log_metadata,
    std::function<void(const std::shared_ptr<RequestLogger>&)> fn) {
  const string& model_name = log_metadata.model_spec().name();
  auto model_to_loggers_map = model_to_loggers_map_.get();
  if (!model_to_loggers_map || model_to_loggers_map->empty()) {
    VLOG(2) << "Request loggers map is empty.";
    return;
  }
  auto found_it = model_to_loggers_map->find(model_name);
  if (found_it == model_to_loggers_map->end()) {
    VLOG(2) << "Cannot find request-loggers for model: " << model_name;
    return;
  }
  for (const auto& logger : found_it->second) {
    fn(logger);
  }
}

}  // namespace serving
}  // namespace tensorflow
