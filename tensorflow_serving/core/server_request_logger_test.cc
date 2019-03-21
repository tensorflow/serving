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

#include "tensorflow_serving/core/server_request_logger.h"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/message.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/config/log_collector_config.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/log_collector.h"
#include "tensorflow_serving/core/logging.pb.h"
#include "tensorflow_serving/core/test_util/fake_log_collector.h"
#include "tensorflow_serving/core/test_util/mock_request_logger.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::NiceMock;

LogCollectorConfig CreateLogCollectorConfig(const string& type,
                                            const string& filename_prefix) {
  LogCollectorConfig config;
  config.set_type(type);
  config.set_filename_prefix(filename_prefix);
  return config;
}

std::pair<string, LoggingConfig> CreateLoggingConfigForModel(
    const string& model_name, const string& log_filename_suffix = "") {
  string filename_prefix = absl::StrCat("/file/", model_name);
  if (!log_filename_suffix.empty()) {
    absl::StrAppend(&filename_prefix, "-", log_filename_suffix);
  }
  LoggingConfig logging_config;
  *logging_config.mutable_log_collector_config() =
      CreateLogCollectorConfig("", filename_prefix);
  logging_config.mutable_sampling_config()->set_sampling_rate(1.0);
  return {model_name, logging_config};
}

std::map<string, std::vector<LoggingConfig>> CreateLoggingConfigMap(
    const std::vector<std::pair<string, LoggingConfig>>& model_configs) {
  std::map<string, std::vector<LoggingConfig>> config_map;
  for (const auto& model_config : model_configs) {
    config_map[model_config.first].push_back(model_config.second);
  }
  return config_map;
}

class ServerRequestLoggerTest : public ::testing::Test {
 protected:
  ServerRequestLoggerTest() {
    TF_CHECK_OK(ServerRequestLogger::Create(
        [&](const LoggingConfig& logging_config,
            std::unique_ptr<RequestLogger>* const request_logger) {
          const string& filename_prefix =
              logging_config.log_collector_config().filename_prefix();
          log_collector_map_[filename_prefix] = new FakeLogCollector();
          increment_created_logger_counter();
          auto logger_destruction_notifier = [this]() {
            increment_deleted_logger_counter();
          };
          const std::vector<string>& tags = {kSavedModelTagServe};
          auto mock_request_logger =
              std::unique_ptr<NiceMock<MockRequestLogger>>(
                  new NiceMock<MockRequestLogger>(
                      logging_config, tags, log_collector_map_[filename_prefix],
                      logger_destruction_notifier));
          ON_CALL(*mock_request_logger, CreateLogMessage(_, _, _, _))
              .WillByDefault(Invoke([&](const google::protobuf::Message& actual_request,
                                        const google::protobuf::Message& actual_response,
                                        const LogMetadata& actual_log_metadata,
                                        std::unique_ptr<google::protobuf::Message>* log) {
                *log = std::unique_ptr<google::protobuf::Any>(
                    new google::protobuf::Any());
                return request_logger_status_cb_();
              }));
          *request_logger = std::move(mock_request_logger);
          return Status::OK();
        },
        &server_request_logger_));
  }

  void increment_created_logger_counter() {
    mutex_lock l(m_);
    created_logger_counter_++;
  }

  int created_logger_counter() const {
    mutex_lock sl(m_);
    return created_logger_counter_;
  }

  void increment_deleted_logger_counter() {
    mutex_lock l(m_);
    deleted_logger_counter_++;
  }

  int deleted_logger_counter() const {
    mutex_lock sl(m_);
    return deleted_logger_counter_;
  }

  mutable mutex m_;
  int created_logger_counter_ = 0;
  int deleted_logger_counter_ = 0;
  std::function<Status()> request_logger_status_cb_ = []() {
    return Status::OK();
  };
  std::unordered_map<string, FakeLogCollector*> log_collector_map_;
  std::unique_ptr<ServerRequestLogger> server_request_logger_;
};

TEST_F(ServerRequestLoggerTest, Empty) {
  TF_ASSERT_OK(server_request_logger_->Update({}));
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           LogMetadata()));
  // No log-collectors should have been made.
  EXPECT_TRUE(log_collector_map_.empty());
}

TEST_F(ServerRequestLoggerTest, AbsentModel) {
  TF_ASSERT_OK(server_request_logger_->Update(
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0")})));
  LogMetadata log_metadata;
  auto* const model_spec = log_metadata.mutable_model_spec();
  model_spec->set_name("absent_model");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata));
  ASSERT_EQ(1, log_collector_map_.size());
  EXPECT_EQ(0, log_collector_map_["/file/model0"]->collect_count());
}

TEST_F(ServerRequestLoggerTest, MultipleModels) {
  TF_ASSERT_OK(server_request_logger_->Update(
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0"),
                              CreateLoggingConfigForModel("model1")})));

  LogMetadata log_metadata0;
  auto* const model_spec0 = log_metadata0.mutable_model_spec();
  model_spec0->set_name("model0");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata0));
  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0"]->collect_count());
  EXPECT_EQ(0, log_collector_map_["/file/model1"]->collect_count());

  LogMetadata log_metadata1;
  auto* const model_spec = log_metadata1.mutable_model_spec();
  model_spec->set_name("model1");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata1));
  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0"]->collect_count());
  EXPECT_EQ(1, log_collector_map_["/file/model1"]->collect_count());
}

TEST_F(ServerRequestLoggerTest, CreateAndDeleteLogger) {
  auto model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0")});
  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  EXPECT_EQ(1, created_logger_counter());
  EXPECT_EQ(0, deleted_logger_counter());

  model_logging_configs.clear();
  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  EXPECT_EQ(1, created_logger_counter());
  EXPECT_EQ(1, deleted_logger_counter());
}

TEST_F(ServerRequestLoggerTest, CreateAndModifyLogger) {
  auto model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0")});
  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  EXPECT_EQ(1, created_logger_counter());
  EXPECT_EQ(0, deleted_logger_counter());

  model_logging_configs["model0"][0]
      .mutable_sampling_config()
      ->set_sampling_rate(0.17);

  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  EXPECT_EQ(2, created_logger_counter());
  EXPECT_EQ(1, deleted_logger_counter());
}

TEST_F(ServerRequestLoggerTest, SameConfigForTwoModelsCreatesOneLogger) {
  std::pair<string, LoggingConfig> model_and_config1 =
      CreateLoggingConfigForModel("model");
  std::pair<string, LoggingConfig> model_and_config2 = {
      "model2", model_and_config1.second};
  TF_ASSERT_OK(server_request_logger_->Update(
      CreateLoggingConfigMap({model_and_config1, model_and_config2})));

  EXPECT_EQ(1, created_logger_counter());
  EXPECT_EQ(0, deleted_logger_counter());
}

TEST_F(ServerRequestLoggerTest, MultipleConfigForOneModel) {
  TF_ASSERT_OK(server_request_logger_->Update(CreateLoggingConfigMap(
      {CreateLoggingConfigForModel("model0"),
       CreateLoggingConfigForModel("model0", "infra")})));
  EXPECT_EQ(2, created_logger_counter());
  EXPECT_EQ(0, deleted_logger_counter());
}

TEST_F(ServerRequestLoggerTest, MultipleLoggersForOneModel) {
  TF_ASSERT_OK(server_request_logger_->Update(CreateLoggingConfigMap(
      {CreateLoggingConfigForModel("model0"),
       CreateLoggingConfigForModel("model0", "infra")})));

  LogMetadata log_metadata0;
  auto* const model_spec0 = log_metadata0.mutable_model_spec();
  model_spec0->set_name("model0");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata0));
  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0"]->collect_count());
  EXPECT_EQ(1, log_collector_map_["/file/model0-infra"]->collect_count());

  LogMetadata log_metadata1;
  auto* const model_spec = log_metadata1.mutable_model_spec();
  model_spec->set_name("model1");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata1));
  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0"]->collect_count());
  EXPECT_EQ(1, log_collector_map_["/file/model0-infra"]->collect_count());
  EXPECT_EQ(log_collector_map_.end(), log_collector_map_.find("/file/model1"));
}

TEST_F(ServerRequestLoggerTest, MultipleLoggersOneModelErrors) {
  TF_ASSERT_OK(server_request_logger_->Update(CreateLoggingConfigMap(
      {CreateLoggingConfigForModel("model0"),
       CreateLoggingConfigForModel("model0", "infra")})));

  // Inject errors for all Log() calls.
  int req_count = 0;
  request_logger_status_cb_ = [&]() {
    return errors::InvalidArgument(absl::StrCat(req_count++));
  };

  LogMetadata log_metadata0;
  auto* const model_spec0 = log_metadata0.mutable_model_spec();
  model_spec0->set_name("model0");
  EXPECT_EQ(errors::InvalidArgument("0"),
            server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                        log_metadata0));

  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(0, log_collector_map_["/file/model0"]->collect_count());
  EXPECT_EQ(0, log_collector_map_["/file/model0-infra"]->collect_count());
}

TEST_F(ServerRequestLoggerTest, MultipleUpdatesSingleCreation) {
  const auto& model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0"),
                              CreateLoggingConfigForModel("model1")});
  for (int i = 0; i < 100; i++) {
    TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  }

  EXPECT_EQ(2, created_logger_counter());
  EXPECT_EQ(0, deleted_logger_counter());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
