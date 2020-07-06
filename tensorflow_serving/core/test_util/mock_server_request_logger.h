/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_SERVER_REQUEST_LOGGER_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_SERVER_REQUEST_LOGGER_H_

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "tensorflow_serving/core/server_request_logger.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockServerRequestLogger : public ServerRequestLogger {
 public:
  MockServerRequestLogger() : ServerRequestLogger({}) {}

  MOCK_METHOD(Status, Update,
              ((const std::map<string, std::vector<LoggingConfig>>&
                    logging_config_map)),
              (override));

  MOCK_METHOD(Status, Log,
              (const google::protobuf::Message& request, const google::protobuf::Message& response,
               const LogMetadata& log_metadata),
              (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_SERVER_REQUEST_LOGGER_H_
