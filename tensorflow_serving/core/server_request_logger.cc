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
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/logging.pb.h"

namespace tensorflow {
namespace serving {

namespace {

// Validate the logging config map. For now only check for duplicate filename
// prefixes and emit warnings.
Status ValidateLoggingConfigMap(
    const std::map<string, LoggingConfig>& logging_config_map) {
  std::set<string> filename_prefixes;
  for (const auto& model_and_logging_config : logging_config_map) {
    const string& filename_prefix =
        model_and_logging_config.second.log_collector_config()
            .filename_prefix();
    if (!gtl::InsertIfNotPresent(&filename_prefixes, filename_prefix)) {
      // Each model's logs are supposed to be separated from each other,
      // though there could be systems which can distinguish based on the
      // model-spec in the logging proto, so we issue only a warning.
      LOG(WARNING) << "Duplicate LogCollectorConfig::filename_prefix(): "
                   << filename_prefix << ". Possibly a misconfiguration.";
    }
  }
  return Status::OK();
}

}  // namespace

// static
Status ServerRequestLogger::Create(
    LoggerCreator request_logger_creator,
    std::unique_ptr<ServerRequestLogger>* server_request_logger) {
  server_request_logger->reset(
      new ServerRequestLogger(std::move(request_logger_creator)));
  return Status::OK();
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
    return Status::OK();
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
    return Status::OK();
  }

  // The logger does not exist. Create a new logger, insert it into
  // new_config_to_logger_map and return it.
  std::unique_ptr<RequestLogger> logger;
  TF_RETURN_IF_ERROR(request_logger_creator_(config, &logger));
  *result = logger.get();
  new_config_to_logger_map->emplace(
      std::make_pair(serialized_config, std::move(logger)));
  return Status::OK();
}

Status ServerRequestLogger::Update(
    const std::map<string, LoggingConfig>& logging_config_map) {
  if (!logging_config_map.empty() && !request_logger_creator_) {
    return errors::InvalidArgument("No request-logger-creator provided.");
  }
  TF_RETURN_IF_ERROR(ValidateLoggingConfigMap(logging_config_map));

  // Those new maps will only contain loggers from logging_config_map and
  // replace the current versions further down.
  std::unique_ptr<StringToRequestLoggerMap> new_model_to_logger_map(
      new StringToRequestLoggerMap());
  StringToUniqueRequestLoggerMap new_config_to_logger_map;

  mutex_lock l(update_mu_);

  for (const auto& model_and_logging_config : logging_config_map) {
    RequestLogger* logger;
    TF_RETURN_IF_ERROR(FindOrCreateLogger(model_and_logging_config.second,
                                          &new_config_to_logger_map, &logger));
    const string& model_name = model_and_logging_config.first;
    new_model_to_logger_map->emplace(std::make_pair(model_name, logger));
  }

  model_to_logger_map_.Update(std::move(new_model_to_logger_map));
  // Any remaining loggers in config_to_logger_map_ will not be needed anymore
  // and destructed at this point.
  config_to_logger_map_ = std::move(new_config_to_logger_map);

  return Status::OK();
}

Status ServerRequestLogger::Log(const google::protobuf::Message& request,
                                const google::protobuf::Message& response,
                                const LogMetadata& log_metadata) {
  const string& model_name = log_metadata.model_spec().name();
  auto model_to_logger_map = model_to_logger_map_.get();
  if (!model_to_logger_map || model_to_logger_map->empty()) {
    VLOG(2) << "Request logger map is empty.";
    return Status::OK();
  }
  auto found_it = model_to_logger_map->find(model_name);
  if (found_it == model_to_logger_map->end()) {
    VLOG(2) << "Cannot find request-logger for model: " << model_name;
    return Status::OK();
  }
  auto& request_logger = found_it->second;
  return request_logger->Log(request, response, log_metadata);
}

}  // namespace serving
}  // namespace tensorflow
