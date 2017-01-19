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
  // Creates the ServerRequestLogger based on a custom request_logger_creator
  // method.
  //
  // You can create an empty ServerRequestLogger with an empty
  // request_logger_creator.
  static Status Create(
      const std::function<Status(const LoggingConfig& logging_config,
                                 std::unique_ptr<RequestLogger>*)>&
          request_logger_creator,
      std::unique_ptr<ServerRequestLogger>* server_request_logger);

  ~ServerRequestLogger() = default;

  // Updates the logger with the new 'logging_config_map'.
  //
  // If the ServerRequestLogger was created using an empty
  // request_logger_creator, this will return an error if a non-empty
  // logging_config_map is passed in.
  Status Update(const std::map<string, LoggingConfig>& logging_config_map);

  // Similar to RequestLogger::Log().
  Status Log(const google::protobuf::Message& request, const google::protobuf::Message& response,
             const LogMetadata& log_metadata);

 private:
  explicit ServerRequestLogger(
      const std::function<Status(const LoggingConfig& logging_config,
                                 std::unique_ptr<RequestLogger>*)>&
          request_logger_creator);

  // A map from model_name to its corresponding RequestLogger.
  using RequestLoggerMap =
      std::unordered_map<string, std::unique_ptr<RequestLogger>>;
  FastReadDynamicPtr<RequestLoggerMap> request_logger_map_;

  std::function<Status(const LoggingConfig& logging_config,
                       std::unique_ptr<RequestLogger>*)>
      request_logger_creator_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
