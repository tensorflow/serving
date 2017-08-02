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

#ifndef TENSORFLOW_SERVING_CORE_REQUEST_LOGGER_H_
#define TENSORFLOW_SERVING_CORE_REQUEST_LOGGER_H_

#include <random>

#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/log_collector.h"
#include "tensorflow_serving/core/logging.pb.h"

namespace tensorflow {
namespace serving {

// Abstraction to log requests and responses hitting a server. The log storage
// is handled by the log-collector. We sample requests based on the config.
class RequestLogger {
 public:
  RequestLogger(const LoggingConfig& logging_config,
                std::unique_ptr<LogCollector> log_collector);

  virtual ~RequestLogger() = default;

  // Writes the log for the particular request, respone and metadata, if we
  // decide to sample it.
  Status Log(const google::protobuf::Message& request, const google::protobuf::Message& response,
             const LogMetadata& log_metadata);

  const LoggingConfig& logging_config() const { return logging_config_; }

 private:
  // Creates the log message given the request, response and metadata.
  // Implementations override it to create the particular message that they want
  // to be logged.
  virtual Status CreateLogMessage(const google::protobuf::Message& request,
                                  const google::protobuf::Message& response,
                                  const LogMetadata& log_metadata,
                                  std::unique_ptr<google::protobuf::Message>* log) = 0;

  // A sampler which samples uniformly at random.
  class UniformSampler {
   public:
    UniformSampler() : rd_(), gen_(rd_()), dist_(0, 1) {}

    // Returns true if the sampler decides to sample it with a probability
    // 'rate'.
    bool Sample(const double rate) { return dist_(gen_) < rate; }

   private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist_;
  };

  const LoggingConfig logging_config_;
  std::unique_ptr<LogCollector> log_collector_;
  UniformSampler uniform_sampler_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_REQUEST_LOGGER_H_
