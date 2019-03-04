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

#ifndef TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
#define TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/logging.pb.h"
#include "tensorflow_serving/core/request_logger.h"
#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"

namespace tensorflow {
namespace serving {

// Logs a sample of requests hitting all the models in the server.
//
// Constructed based on the logging config for the server, which contains the
// sampling config.
class ServerRequestLogger {
 public:
  using LoggerCreator = std::function<Status(
      const LoggingConfig& logging_config, std::unique_ptr<RequestLogger>*)>;
  // Creates the ServerRequestLogger based on a custom request_logger_creator
  // method.
  //
  // You can create an empty ServerRequestLogger with an empty
  // request_logger_creator.
  static Status Create(
      LoggerCreator request_logger_creator,
      std::unique_ptr<ServerRequestLogger>* server_request_logger);

  virtual ~ServerRequestLogger() = default;

  // Updates the logger with the new 'logging_config_map'.
  //
  // If the ServerRequestLogger was created using an empty
  // request_logger_creator, this will return an error if a non-empty
  // logging_config_map is passed in.
  virtual Status Update(
      const std::map<string, std::vector<LoggingConfig>>& logging_config_map);

  // Similar to RequestLogger::Log().
  //
  // If request is logged/written to multiple sinks, we return error from
  // the first failed write (and continue attempting to write to all).
  virtual Status Log(const google::protobuf::Message& request,
                     const google::protobuf::Message& response,
                     const LogMetadata& log_metadata);

 protected:
  explicit ServerRequestLogger(LoggerCreator request_logger_creator);

 private:
  using StringToRequestLoggersMap =
      std::unordered_map<string, std::vector<RequestLogger*>>;
  using StringToUniqueRequestLoggerMap =
      std::unordered_map<string, std::unique_ptr<RequestLogger>>;

  // Find a logger for config in either config_to_logger_map_ or
  // new_config_to_logger_map. If the logger was found in
  // config_to_logger_map_ move it to new_config_to_logger_map and erase the
  // entry from config_to_logger_map_. If such a logger does not exist,
  // create a new logger and insert it into new_config_to_logger_map. Return the
  // logger in result.
  Status FindOrCreateLogger(
      const LoggingConfig& config,
      StringToUniqueRequestLoggerMap* new_config_to_logger_map,
      RequestLogger** result);

  // Mutex to ensure concurrent calls to Update() are serialized.
  mutable mutex update_mu_;
  // A map from serialized model logging config to its corresponding
  // RequestLogger. If two models have the same logging config, they
  // will share the RequestLogger.
  // This is only used during calls to Update().
  StringToUniqueRequestLoggerMap config_to_logger_map_;

  // A map from model_name to its corresponding RequestLoggers.
  // The RequestLoggers are owned by config_to_logger_map_.
  FastReadDynamicPtr<StringToRequestLoggersMap> model_to_loggers_map_;

  LoggerCreator request_logger_creator_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
