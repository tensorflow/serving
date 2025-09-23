/* Copyright 2023 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_PREDICTION_STREAM_LOGGER_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_PREDICTION_STREAM_LOGGER_H_

#include <memory>

#include <gmock/gmock.h>
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/stream_logger.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockPredictionStreamLogger
    : public StreamLogger<PredictRequest, PredictResponse> {
 public:
  ~MockPredictionStreamLogger() override = default;

  MOCK_METHOD(void, LogStreamRequest, (PredictRequest), (override));
  MOCK_METHOD(void, LogStreamResponse, (PredictResponse), (override));
  MOCK_METHOD(absl::Status, CreateLogMessage,
              (const LogMetadata&, std::unique_ptr<google::protobuf::Message>*),
              (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_PREDICTION_STREAM_LOGGER_H_
