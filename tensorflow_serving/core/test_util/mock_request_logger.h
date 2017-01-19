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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_

#include "google/protobuf/message.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/log_collector.h"
#include "tensorflow_serving/core/logging.pb.h"
#include "tensorflow_serving/core/request_logger.h"

namespace tensorflow {
namespace serving {

class MockRequestLogger : public RequestLogger {
 public:
  // Unfortunately NiceMock doesn't support ctors with move-only types, so we
  // have to do this workaround.
  MockRequestLogger(const LoggingConfig& logging_config,
                    LogCollector* log_collector)
      : RequestLogger(logging_config,
                      std::unique_ptr<LogCollector>(log_collector)) {}

  MOCK_METHOD4(CreateLogMessage, Status(const google::protobuf::Message& request,
                                        const google::protobuf::Message& response,
                                        const LogMetadata& log_metadata,
                                        std::unique_ptr<google::protobuf::Message>* log));
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_REQUEST_LOGGER_H_
