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

#include "tensorflow_serving/core/request_logger.h"

#include <memory>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "google/protobuf/message.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/log_collector.h"
#include "tensorflow_serving/core/logging.pb.h"
#include "tensorflow_serving/core/test_util/mock_log_collector.h"
#include "tensorflow_serving/core/test_util/mock_request_logger.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;

class RequestLoggerTest : public ::testing::Test {
 protected:
  RequestLoggerTest() {
    LoggingConfig logging_config;
    logging_config.mutable_sampling_config()->set_sampling_rate(1.0);
    log_collector_ = new NiceMock<MockLogCollector>();
    request_logger_ = std::unique_ptr<NiceMock<MockRequestLogger>>(
        new NiceMock<MockRequestLogger>(logging_config, log_collector_));
  }

  NiceMock<MockLogCollector>* log_collector_;
  std::unique_ptr<NiceMock<MockRequestLogger>> request_logger_;
};

TEST_F(RequestLoggerTest, Simple) {
  ModelSpec model_spec;
  model_spec.set_name("model");
  model_spec.mutable_version()->set_value(10);

  PredictRequest request;
  *request.mutable_model_spec() = model_spec;

  PredictResponse response;
  response.mutable_outputs()->insert({"tensor", TensorProto()});
  LogMetadata log_metadata;
  *log_metadata.mutable_model_spec() = model_spec;

  EXPECT_CALL(*request_logger_, CreateLogMessage(_, _, _, _))
      .WillOnce(Invoke([&](const google::protobuf::Message& actual_request,
                           const google::protobuf::Message& actual_response,
                           const LogMetadata& actual_log_metadata,
                           std::unique_ptr<google::protobuf::Message>* log) {
        EXPECT_THAT(static_cast<const PredictRequest&>(actual_request),
                    test_util::EqualsProto(request));
        EXPECT_THAT(static_cast<const PredictResponse&>(actual_response),
                    test_util::EqualsProto(PredictResponse()));
        LogMetadata expected_log_metadata = log_metadata;
        expected_log_metadata.mutable_sampling_config()->set_sampling_rate(1.0);
        EXPECT_THAT(actual_log_metadata,
                    test_util::EqualsProto(expected_log_metadata));
        *log =
            std::unique_ptr<google::protobuf::Any>(new google::protobuf::Any());
        return Status::OK();
      }));
  EXPECT_CALL(*log_collector_, CollectMessage(_))
      .WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(request_logger_->Log(request, PredictResponse(), log_metadata));
}

TEST_F(RequestLoggerTest, ErroringCreateLogMessage) {
  EXPECT_CALL(*request_logger_, CreateLogMessage(_, _, _, _))
      .WillRepeatedly(Return(errors::Internal("Error")));
  EXPECT_CALL(*log_collector_, CollectMessage(_)).Times(0);
  const auto error_status =
      request_logger_->Log(PredictRequest(), PredictResponse(), LogMetadata());
  ASSERT_FALSE(error_status.ok());
  EXPECT_THAT(error_status.error_message(), HasSubstr("Error"));
}

TEST_F(RequestLoggerTest, ErroringCollectMessage) {
  EXPECT_CALL(*request_logger_, CreateLogMessage(_, _, _, _))
      .WillRepeatedly(Invoke([&](const google::protobuf::Message& actual_request,
                                 const google::protobuf::Message& actual_response,
                                 const LogMetadata& actual_log_metadata,
                                 std::unique_ptr<google::protobuf::Message>* log) {
        *log =
            std::unique_ptr<google::protobuf::Any>(new google::protobuf::Any());
        return Status::OK();
      }));
  EXPECT_CALL(*log_collector_, CollectMessage(_))
      .WillRepeatedly(Return(errors::Internal("Error")));
  const auto error_status =
      request_logger_->Log(PredictRequest(), PredictResponse(), LogMetadata());
  ASSERT_FALSE(error_status.ok());
  EXPECT_THAT(error_status.error_message(), HasSubstr("Error"));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
