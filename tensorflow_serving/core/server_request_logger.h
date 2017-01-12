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

namespace tensorflow {
namespace serving {

// Logs a sample of requests hitting all the models in the server.
//
// Constructed based on the logging config for the server, which contains the
// sampling config.
class ServerRequestLogger {
 public:
  // Creates the ServerRequestLogger based on the map of logging-configs keyed
  // on model-names, and a custom request-logger creator method.
  static Status Create(
      const std::map<string, LoggingConfig>& logging_config_map,
      const std::function<Status(const LogCollectorConfig& log_collector_config,
                                 std::unique_ptr<RequestLogger>*)>&
          request_logger_creator,
      std::unique_ptr<ServerRequestLogger>* server_request_logger);

  ~ServerRequestLogger() = default;

  // Similar to RequestLogger::Log().
  Status Log(const google::protobuf::Message& request, const google::protobuf::Message& response,
             const LogMetadata& log_metadata);

 private:
  explicit ServerRequestLogger(
      std::unordered_map<string, std::unique_ptr<RequestLogger>>
          request_logger_map);

  // A map from model-name to its corresponding request-logger.
  std::unordered_map<string, std::unique_ptr<RequestLogger>>
      request_logger_map_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
