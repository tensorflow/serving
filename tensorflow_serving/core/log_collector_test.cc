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

#include "tensorflow_serving/core/log_collector.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/config/log_collector_config.pb.h"

namespace tensorflow {
namespace serving {
namespace {

class FakeLogCollector : public LogCollector {
 public:
  Status CollectMessage(const google::protobuf::Message& message) { return Status::OK(); }
  Status Flush() override { return Status::OK(); }
};

LogCollectorConfig CreateConfig(const string& type,
                                const string& filename_prefix) {
  LogCollectorConfig config;
  config.set_type(type);
  config.set_filename_prefix(filename_prefix);
  return config;
}

TEST(LogCollectorTest, NotRegistered) {
  std::unique_ptr<LogCollector> log_collector;
  const auto status = LogCollector::Create(
      CreateConfig("notregistered", "filename_prefix"), 0, &log_collector);
  EXPECT_EQ(status.code(), error::NOT_FOUND);
}

TEST(LogCollectorTest, Registration) {
  TF_ASSERT_OK(LogCollector::RegisterFactory(
      "registered", [](const LogCollectorConfig& config, const uint32 id,
                       std::unique_ptr<LogCollector>* log_collector) {
        *log_collector = std::unique_ptr<LogCollector>(new FakeLogCollector());
        return Status::OK();
      }));
  std::unique_ptr<LogCollector> log_collector;
  TF_ASSERT_OK(LogCollector::Create(
      CreateConfig("registered", "filename_prefix"), 0, &log_collector));
}

auto duplicate_factory = [](const LogCollectorConfig& config, const uint32 id,
                            std::unique_ptr<LogCollector>* log_collector) {
  *log_collector = std::unique_ptr<LogCollector>(new FakeLogCollector());
  return Status::OK();
};
REGISTER_LOG_COLLECTOR("duplicate", duplicate_factory);

TEST(LogCollectorTest, DuplicateRegistration) {
  const auto status = LogCollector::RegisterFactory(
      "duplicate", [](const LogCollectorConfig& config, const uint32 id,
                      std::unique_ptr<LogCollector>* log_collector) {
        *log_collector = std::unique_ptr<LogCollector>(new FakeLogCollector());
        return Status::OK();
      });
  EXPECT_EQ(status.code(), error::ALREADY_EXISTS);
}

auto creation_factory = [](const LogCollectorConfig& config, const uint32 id,
                           std::unique_ptr<LogCollector>* log_collector) {
  *log_collector = std::unique_ptr<LogCollector>(new FakeLogCollector());
  return Status::OK();
};
REGISTER_LOG_COLLECTOR("creation", creation_factory);

TEST(LogCollectorTest, Creation) {
  std::unique_ptr<LogCollector> log_collector0;
  TF_ASSERT_OK(LogCollector::Create(CreateConfig("creation", "filename_prefix"),
                                    0, &log_collector0));
  std::unique_ptr<LogCollector> log_collector1;
  TF_ASSERT_OK(LogCollector::Create(CreateConfig("creation", "filename_prefix"),
                                    0, &log_collector1));
  EXPECT_NE(log_collector0.get(), log_collector1.get());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
