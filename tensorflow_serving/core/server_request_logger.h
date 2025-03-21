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

#ifndef TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
#define TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/logging.pb.h"
#include "tensorflow_serving/config/logging_config.pb.h"
#include "tensorflow_serving/core/request_logger.h"
#include "tensorflow_serving/core/stream_logger.h"
#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"

namespace tensorflow {
namespace serving {

// Logs a sample of requests hitting all the models in the server.
//
// Constructed based on the logging config for the server, which contains the
// sampling config.
class ServerRequestLogger {
 public:
  using LoggerCreator = std::function<absl::Status(
      const LoggingConfig& logging_config, std::shared_ptr<RequestLogger>*)>;

  // Creates the ServerRequestLogger based on a custom request_logger_creator
  // method.
  //
  // You can create an empty ServerRequestLogger with an empty
  // request_logger_creator.
  static absl::Status Create(
      LoggerCreator request_logger_creator,
      std::unique_ptr<ServerRequestLogger>* server_request_logger);

  virtual ~ServerRequestLogger() = default;

  using StringToRequestLoggersMap =
      std::unordered_map<string, std::vector<std::shared_ptr<RequestLogger>>>;
  using StringToUniqueRequestLoggerMap =
      std::unordered_map<string, std::shared_ptr<RequestLogger>>;
  struct UpdateRequest {
    absl::Nonnull<std::unique_ptr<StringToRequestLoggersMap>>
        model_to_loggers_map;
    absl::Nonnull<std::unique_ptr<StringToUniqueRequestLoggerMap>>
        config_to_logger_map;
  };

  // `CreateUpdateRequest()` allows you to validate the given logging config
  // map without actually applying the update. A `UpdateRequest` struct is
  // returned if the given `logging_config_map` passed all the validation
  // checks, and the returned object can be passed to `ApplyUpdateRequest()`
  // function which allows an infallible update later. This allows the caller to
  // perform the validation step and the apply step separately.
  //
  // Note, if you plan to do validation and apply as separate steps, you must
  // lock this object to ensure no `Update()` or `ApplyUpdateRequest()` is
  // called concurrently.
  virtual absl::StatusOr<UpdateRequest> CreateUpdateRequest(
      const std::map<string, std::vector<LoggingConfig>>& logging_config_map)
      const;

  // Infallible update function. The `req` provided in this function is not
  // validated, for this reason the caller is advised to validate the request
  // before calling this function. Most optimally the caller should use the
  // output generated from the `CreateUpdateRequest()` function.
  virtual void ApplyUpdateRequest(UpdateRequest& req);

  // Fallibly updates the logger with the new 'logging_config_map'.
  //
  // If the ServerRequestLogger was created using an empty
  // request_logger_creator, this will return an error if a non-empty
  // logging_config_map is passed in.
  virtual absl::Status Update(
      const std::map<string, std::vector<LoggingConfig>>& logging_config_map);

  // Similar to RequestLogger::Log().
  //
  // If request is logged/written to multiple sinks, we return error from
  // the first failed write (and continue attempting to write to all).
  virtual absl::Status Log(const google::protobuf::Message& request,
                           const google::protobuf::Message& response,
                           const LogMetadata& log_metadata);

  // Starts logging a stream. Returns a StreamLogger created through
  // `create_stream_logger_fn`. Returns NULL if the stream should not be logged.
  //
  // Even if the stream is logged/written to multiple sinks, we return a single
  // stream logger but register multiple callbacks, one for each sink.
  template <typename Request, typename Response>
  using CreateStreamLoggerFn =
      std::function<std::unique_ptr<StreamLogger<Request, Response>>()>;
  template <typename Request, typename Response>
  std::unique_ptr<StreamLogger<Request, Response>> StartLoggingStream(
      const LogMetadata& log_metadata,
      CreateStreamLoggerFn<Request, Response> create_stream_logger_fn);

 protected:
  explicit ServerRequestLogger(LoggerCreator request_logger_creator);

 private:
  // Locked version of `CreateUpdateRequest()`. `update_mu_` must be held when
  // calling this function.
  absl::StatusOr<UpdateRequest> CreateUpdateRequestLocked(
      const std::map<string, std::vector<LoggingConfig>>& logging_config_map)
      const;

  // Locked version of `ApplyUpdateRequest()`. `update_mu_` must be held when
  // calling this function.
  void ApplyUpdateRequestLocked(UpdateRequest& req);

  // Find a logger for config in either config_to_logger_map_ or
  // new_config_to_logger_map. If the logger was found in one of existing
  // maps, a shared_ptr that points to the existing logging config will be
  // added to the new_config_to_logger_map, this avoids unnecessary creation
  // of duplicate configs. If such a logger does not exist, create a new
  // logger and insert it into new_config_to_logger_map. Return the logger
  // in `result`.
  absl::Status FindOrCreateLogger(
      const LoggingConfig& config,
      StringToUniqueRequestLoggerMap* new_config_to_logger_map,
      std::shared_ptr<RequestLogger>* result) const;

  // Invokes `fn` with all loggers for the corresponding model.
  void InvokeLoggerForModel(
      const LogMetadata& log_metadata,
      std::function<void(const std::shared_ptr<RequestLogger>&)> fn);

  // Mutex to ensure concurrent calls to Update() are serialized.
  mutable absl::Mutex update_mu_;
  // A map from serialized model logging config to its corresponding
  // RequestLogger. If two models have the same logging config, they
  // will share the RequestLogger.
  // This is only used during calls to Update().
  StringToUniqueRequestLoggerMap config_to_logger_map_;

  // A map from model_name to its corresponding RequestLoggers.
  // The RequestLoggers are owned by config_to_logger_map_.
  FastReadDynamicPtr<StringToRequestLoggersMap> model_to_loggers_map_;

  LoggerCreator request_logger_creator_;
};

/***************************Implementation Details***************************/
template <typename Request, typename Response>
std::unique_ptr<StreamLogger<Request, Response>>
ServerRequestLogger::StartLoggingStream(
    const LogMetadata& log_metadata,
    CreateStreamLoggerFn<Request, Response> create_stream_logger_fn) {
  StreamLogger<Request, Response>* stream_logger = nullptr;
  InvokeLoggerForModel(
      log_metadata, [create_stream_logger_fn, &stream_logger, &log_metadata](
                        const std::shared_ptr<RequestLogger>& request_logger) {
        RequestLogger::GetStreamLoggerFn<Request, Response> create_logger =
            [&create_stream_logger_fn, &stream_logger]() {
              stream_logger = create_stream_logger_fn().release();
              return stream_logger;
            };
        RequestLogger::GetStreamLoggerFn<Request, Response> get_logger =
            [stream_logger]() { return stream_logger; };
        request_logger->MaybeStartLoggingStream(
            log_metadata, stream_logger == nullptr ? std::move(create_logger)
                                                   : std::move(get_logger));
        // If the stream logger has been created, reuse it.
      });
  return absl::WrapUnique(stream_logger);
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVER_REQUEST_LOGGER_H_
