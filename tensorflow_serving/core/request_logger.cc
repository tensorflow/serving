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

#include "tensorflow_serving/core/request_logger.h"

#include <random>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow_serving/apis/model.pb.h"

namespace tensorflow {
namespace serving {
namespace {

auto* request_log_count = monitoring::Counter<2>::New(
    "/tensorflow/serving/request_log_count",
    "The total number of requests logged from the model server sliced "
    "down by model_name and status code.",
    "model_name", "status_code");
}

RequestLogger::RequestLogger(const LoggingConfig& logging_config,
                             const std::vector<string>& saved_model_tags,
                             std::unique_ptr<LogCollector> log_collector)
    : logging_config_(logging_config),
      saved_model_tags_(saved_model_tags),
      log_collector_(std::move(log_collector)),
      uniform_sampler_() {}

Status RequestLogger::Log(const google::protobuf::Message& request,
                          const google::protobuf::Message& response,
                          const LogMetadata& log_metadata) {
  const double sampling_rate =
      logging_config_.sampling_config().sampling_rate();
  LogMetadata log_metadata_with_config = log_metadata;
  *log_metadata_with_config.mutable_sampling_config() =
      logging_config_.sampling_config();
  if (!saved_model_tags_.empty()) {
    *log_metadata_with_config.mutable_saved_model_tags() = {
        saved_model_tags_.begin(), saved_model_tags_.end()};
  }
  if (uniform_sampler_.Sample(sampling_rate)) {
    const auto status = [&]() {
      std::unique_ptr<google::protobuf::Message> log;
      TF_RETURN_IF_ERROR(
          CreateLogMessage(request, response, log_metadata_with_config, &log));
      return log_collector_->CollectMessage(*log);
    }();
    request_log_count
        ->GetCell(log_metadata.model_spec().name(),
                  error::Code_Name(status.code()))
        ->IncrementBy(1);
    return status;
  }
  return OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
