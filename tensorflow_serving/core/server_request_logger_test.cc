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
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/config/log_collector_config.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/request_logger.h"
#include "tensorflow_serving/core/stream_logger.h"
#include "tensorflow_serving/core/test_util/fake_log_collector.h"
#include "tensorflow_serving/core/test_util/mock_prediction_stream_logger.h"
#include "tensorflow_serving/core/test_util/mock_request_logger.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::MockPredictionStreamLogger;
using ::testing::_;
using ::testing::Eq;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::StatusIs;

using UpdateRequest = ServerRequestLogger::UpdateRequest;

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
            std::shared_ptr<RequestLogger>* const request_logger) {
          if (logging_config.has_sampling_config() &&
              logging_config.sampling_config().sampling_rate() < 0) {
            return absl::InvalidArgumentError(
                "Negative log sampling rate provided.");
          }

          const string& filename_prefix =
              logging_config.log_collector_config().filename_prefix();
          log_collector_map_[filename_prefix] = new FakeLogCollector();
          increment_created_logger_counter();
          auto logger_destruction_notifier = [this]() {
            increment_deleted_logger_counter();
          };
          const std::vector<string>& tags = {kSavedModelTagServe};
          auto mock_request_logger =
              std::shared_ptr<NiceMock<MockRequestLogger>>(
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
          ON_CALL(*mock_request_logger, FillLogMetadata(_))
              .WillByDefault(Invoke([&](const LogMetadata& log_metadata) {
                return log_metadata;
              }));
          *request_logger = std::move(mock_request_logger);
          return absl::OkStatus();
        },
        &server_request_logger_));
  }

  void increment_created_logger_counter() {
    absl::MutexLock l(&m_);
    created_logger_counter_++;
  }

  int created_logger_counter() const {
    absl::MutexLock l(&m_);
    return created_logger_counter_;
  }

  void increment_deleted_logger_counter() {
    absl::MutexLock l(&m_);
    deleted_logger_counter_++;
  }

  int deleted_logger_counter() const {
    absl::MutexLock l(&m_);
    return deleted_logger_counter_;
  }

  mutable absl::Mutex m_;
  int created_logger_counter_ = 0;
  int deleted_logger_counter_ = 0;
  std::function<absl::Status()> request_logger_status_cb_ = []() {
    return absl::OkStatus();
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

TEST_F(ServerRequestLoggerTest, PartiallyBadUpdate) {
  // Initially create a logger with 2 OK configs.
  std::pair<string, LoggingConfig> model0_ok_config =
      CreateLoggingConfigForModel(/*model_name=*/"model0",
                                  /*log_filename_suffix=*/"path0");
  std::pair<string, LoggingConfig> model1_ok_config =
      CreateLoggingConfigForModel(/*model_name=*/"model1",
                                  /*log_filename_suffix=*/"path1");
  TF_ASSERT_OK(server_request_logger_->Update(
      CreateLoggingConfigMap({model0_ok_config, model1_ok_config})));
  EXPECT_EQ(created_logger_counter(), 2);
  EXPECT_EQ(deleted_logger_counter(), 0);

  // Now, attempt to update model1's config with a bad config, expect an update
  // failure, and existing ok configs should not modified.
  std::pair<string, LoggingConfig> model1_bad_config =
      CreateLoggingConfigForModel(/*model_name=*/"model1",
                                  /*log_filename_suffix=*/"path2");
  model1_bad_config.second.mutable_sampling_config()->set_sampling_rate(-1);
  EXPECT_THAT(server_request_logger_->Update(CreateLoggingConfigMap(
                  {model0_ok_config, model1_bad_config})),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(created_logger_counter(), 2);
  EXPECT_EQ(deleted_logger_counter(), 0);

  // Model0 is still using the old config.
  LogMetadata log_metadata0;
  log_metadata0.mutable_model_spec()->set_name("model0");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata0));
  ASSERT_EQ(log_collector_map_.size(), 2);
  EXPECT_EQ(log_collector_map_.count("/file/model0-path0"), 1);
  EXPECT_EQ(log_collector_map_["/file/model0-path0"]->collect_count(), 1);

  // Model1 is still using the old config.
  LogMetadata log_metadata1;
  log_metadata1.mutable_model_spec()->set_name("model1");
  TF_ASSERT_OK(server_request_logger_->Log(PredictRequest(), PredictResponse(),
                                           log_metadata1));
  EXPECT_EQ(log_collector_map_.count("/file/model1-path1"), 1);
  EXPECT_EQ(log_collector_map_["/file/model1-path1"]->collect_count(), 1);
  EXPECT_EQ(log_collector_map_.count("/file/model1-path2"), 0);
}

TEST_F(ServerRequestLoggerTest, CreateUpdateRequestErrors) {
  // Null.
  TF_ASSERT_OK_AND_ASSIGN(UpdateRequest req,
                          server_request_logger_->CreateUpdateRequest({}));
  EXPECT_TRUE(req.config_to_logger_map->empty());
  EXPECT_TRUE(req.model_to_loggers_map->empty());

  // Bad config.
  std::pair<string, LoggingConfig> model0_bad_config =
      CreateLoggingConfigForModel(/*model_name=*/"model0",
                                  /*log_filename_suffix=*/"bad_config_path");
  model0_bad_config.second.mutable_sampling_config()->set_sampling_rate(-1);
  EXPECT_THAT(server_request_logger_->CreateUpdateRequest(
                  CreateLoggingConfigMap({model0_bad_config})),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Ok+bad config.
  std::pair<string, LoggingConfig> model0_ok_config =
      CreateLoggingConfigForModel(/*model_name=*/"model0",
                                  /*log_filename_suffix=*/"ok_conig_path");
  EXPECT_THAT(
      server_request_logger_->CreateUpdateRequest(
          CreateLoggingConfigMap({model0_ok_config, model0_bad_config})),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ServerRequestLoggerTest, CreateUpdateRequestSingleConfigOk) {
  std::pair<string, LoggingConfig> model0_ok_config =
      CreateLoggingConfigForModel(/*model_name=*/"model0",
                                  /*log_filename_suffix=*/"ok_conig_path");
  TF_ASSERT_OK_AND_ASSIGN(UpdateRequest req,
                          server_request_logger_->CreateUpdateRequest(
                              CreateLoggingConfigMap({model0_ok_config})));
  EXPECT_THAT(*req.model_to_loggers_map,
              UnorderedElementsAre(Pair(Eq("model0"), SizeIs(1))));
  EXPECT_THAT(*req.config_to_logger_map, SizeIs(1));
}

TEST_F(ServerRequestLoggerTest, CreateUpdateRequestMultipleConfigsOk) {
  // Both model0 and model1 are using the same config, expect only 1 logger
  // created.
  std::pair<string, LoggingConfig> model0_v1_config =
      CreateLoggingConfigForModel(/*model_name=*/"model0",
                                  /*log_filename_suffix=*/"same_conig_path");
  std::pair<string, LoggingConfig> model0_v2_config = model0_v1_config;
  model0_v2_config.first = "model0_v2";

  TF_ASSERT_OK_AND_ASSIGN(
      UpdateRequest req,
      server_request_logger_->CreateUpdateRequest(
          CreateLoggingConfigMap({model0_v1_config, model0_v2_config})));
  EXPECT_THAT(*req.model_to_loggers_map,
              UnorderedElementsAre(Pair(Eq("model0"), SizeIs(1)),
                                   Pair(Eq("model0_v2"), SizeIs(1))));

  // Only 1 creation and 1 entry in the config map. As the same config is
  // reused.
  EXPECT_THAT(*req.config_to_logger_map, SizeIs(1));
  EXPECT_EQ(created_logger_counter(), 1);
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

TEST_F(ServerRequestLoggerTest, StreamLoggingBasic) {
  auto model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0", "file1"),
                              CreateLoggingConfigForModel("model0", "file2")});

  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  LogMetadata log_metadata;
  log_metadata.mutable_model_spec()->set_name("model0");
  auto* logger_ptr = new MockPredictionStreamLogger();
  auto logger = server_request_logger_
                    ->StartLoggingStream<PredictRequest, PredictResponse>(
                        log_metadata, [logger_ptr]() {
                          return absl::WrapUnique(logger_ptr);
                        });
  EXPECT_CALL(*logger_ptr, CreateLogMessage(_, _))
      .Times(2)
      .WillRepeatedly(Invoke([](const LogMetadata& log_metadata,
                                std::unique_ptr<google::protobuf::Message>* log) {
        *log = std::make_unique<google::protobuf::Any>();
        return absl::OkStatus();
      }));
  TF_ASSERT_OK(logger->LogMessage());
  ASSERT_EQ(2, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0-file1"]->collect_count());
  EXPECT_EQ(1, log_collector_map_["/file/model0-file2"]->collect_count());
}

TEST_F(ServerRequestLoggerTest, StreamLoggingUpdateLoggingConfig) {
  auto model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0", "file1"),
                              CreateLoggingConfigForModel("model0", "file2")});

  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  LogMetadata log_metadata;
  log_metadata.mutable_model_spec()->set_name("model0");
  auto* logger_ptr = new MockPredictionStreamLogger();
  auto logger = server_request_logger_->StartLoggingStream<PredictRequest,
                                                           PredictResponse>(
      log_metadata, [logger_ptr]() {
        return std::unique_ptr<StreamLogger<PredictRequest, PredictResponse>>(
            logger_ptr);
      });

  model_logging_configs =
      CreateLoggingConfigMap({CreateLoggingConfigForModel("model0", "file2"),
                              CreateLoggingConfigForModel("model0", "file3")});

  // Updates to a new logging config. Since the stream hasn't finished, the
  // stream logger will not use the new config.
  TF_ASSERT_OK(server_request_logger_->Update(model_logging_configs));
  EXPECT_CALL(*logger_ptr, CreateLogMessage(_, _))
      .Times(2)
      .WillRepeatedly(Invoke([](const LogMetadata& log_metadata,
                                std::unique_ptr<google::protobuf::Message>* log) {
        *log =
            std::unique_ptr<google::protobuf::Any>(new google::protobuf::Any());
        return absl::OkStatus();
      }));
  TF_ASSERT_OK(logger->LogMessage());
  ASSERT_EQ(3, log_collector_map_.size());
  EXPECT_EQ(1, log_collector_map_["/file/model0-file2"]->collect_count());
  EXPECT_EQ(0, log_collector_map_["/file/model0-file3"]->collect_count());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
