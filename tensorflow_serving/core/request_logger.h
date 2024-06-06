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
#include <vector>

#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/log_collector.h"
#include "tensorflow_serving/core/stream_logger.h"

namespace tensorflow {
namespace serving {

// Abstraction to log requests and responses hitting a server. The log storage
// is handled by the log-collector. We sample requests based on the config.
// All subclasses must only implement a factory method that returns a
// shared_ptr.
class RequestLogger : public std::enable_shared_from_this<RequestLogger> {
 public:
  RequestLogger(const LoggingConfig& logging_config,
                const std::vector<string>& saved_model_tags,
                std::unique_ptr<LogCollector> log_collector);

  virtual ~RequestLogger() = default;

  // Writes the log for the particular request, response and metadata, if we
  // decide to sample it.
  Status Log(const google::protobuf::Message& request, const google::protobuf::Message& response,
             const LogMetadata& log_metadata);

  // Starts logging a stream through returning a StreamLogger through
  // `get_stream_logger_fn` and registers a log callback. Returns NULL if the
  // stream should not be logged.
  template <typename Request, typename Response>
  using GetStreamLoggerFn = std::function<StreamLogger<Request, Response>*()>;
  template <typename Request, typename Response>
  void MaybeStartLoggingStream(
      const LogMetadata& log_metadata,
      GetStreamLoggerFn<Request, Response> get_stream_logger_fn);

  const LoggingConfig& logging_config() const { return logging_config_; }

 private:
  // Creates the log message given the request, response and metadata.
  // Implementations override it to create the particular message that they want
  // to be logged.
  virtual Status CreateLogMessage(const google::protobuf::Message& request,
                                  const google::protobuf::Message& response,
                                  const LogMetadata& log_metadata,
                                  std::unique_ptr<google::protobuf::Message>* log) = 0;

  // Implementations can fill up additional information to LogMetadata.
  virtual LogMetadata FillLogMetadata(const LogMetadata& lm_in) = 0;

  // Writes the log.
  Status Log(const google::protobuf::Message& log);

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
  const std::vector<string> saved_model_tags_;
  std::unique_ptr<LogCollector> log_collector_;
  UniformSampler uniform_sampler_;
};

/**************************Implementation Detail******************************/
template <typename Request, typename Response>
void RequestLogger::MaybeStartLoggingStream(
    const LogMetadata& log_metadata,
    GetStreamLoggerFn<Request, Response> get_stream_logger_fn) {
  // Sampling happens at the beginning of logging to avoid logging overhead.
  // if request logger goes away during a stream which could happen due to
  // loggin config update, the stream won't be logged.
  if (!uniform_sampler_.Sample(
          logging_config_.sampling_config().sampling_rate())) {
    return;
  }

  auto* stream_logger = get_stream_logger_fn();
  if (stream_logger == nullptr) return;

  LogMetadata lm_out = FillLogMetadata(log_metadata);
  std::weak_ptr<RequestLogger> logger_ref(shared_from_this());
  stream_logger->AddLogCallback(
      lm_out, [logger_ref = std::move(logger_ref)](const google::protobuf::Message& log) {
        // The callback refers back to the request logger. If the
        // request logger goes away after creation but before stream
        // ends, we simply skip this sink.
        if (auto logger = logger_ref.lock(); logger != nullptr) {
          TF_RETURN_IF_ERROR(logger->Log(log));
        }
        return OkStatus();
      });
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_REQUEST_LOGGER_H_
