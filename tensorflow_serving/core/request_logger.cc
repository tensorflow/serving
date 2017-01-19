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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace serving {

RequestLogger::RequestLogger(const LoggingConfig& logging_config,
                             std::unique_ptr<LogCollector> log_collector)
    : logging_config_(logging_config),
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
  if (uniform_sampler_.Sample(sampling_rate)) {
    std::unique_ptr<google::protobuf::Message> log;
    TF_RETURN_IF_ERROR(
        CreateLogMessage(request, response, log_metadata_with_config, &log));
    TF_RETURN_IF_ERROR(log_collector_->CollectMessage(*log));
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
