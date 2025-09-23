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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_CORE_STREAM_LOGGER_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_CORE_STREAM_LOGGER_H_

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/apis/logging.pb.h"

namespace tensorflow {
namespace serving {

// Simple logger for a stream of requests and responses. In practice, the
// lifetime of this class should be attached to the lifetime of a stream.
//
// The class being templated on requests and responses is to avoid RTTI in the
// subclasses.
// Not thread-safe.
template <typename Request, typename Response>
class StreamLogger {
 public:
  StreamLogger() {
    static_assert((std::is_base_of<google::protobuf::Message, Request>::value),
                  "Request must be a proto type.");
    static_assert((std::is_base_of<google::protobuf::Message, Response>::value),
                  "Response must be a proto type.");
  }

  virtual ~StreamLogger() = default;

  virtual void LogStreamRequest(Request request) = 0;
  virtual void LogStreamResponse(Response response) = 0;

  using LogMessageFn = std::function<absl::Status(const google::protobuf::Message&)>;
  // Registers a log callback to be invoked when calling LogMessage();
  void AddLogCallback(const LogMetadata& log_metadata,
                      LogMessageFn log_message_fn);

  // Logs the message with all requests and responses accumulated so far, and
  // invokes all log callbacks sequentially. Upon return, any subsequent calls
  // to any other methods of this class will result in undefined behavior. On
  // multiple callbacks, we return error from the first failed one (and continue
  // attempting the rest).
  absl::Status LogMessage();

 private:
  virtual absl::Status CreateLogMessage(
      const LogMetadata& log_metadata,
      std::unique_ptr<google::protobuf::Message>* log) = 0;

  struct StreamLogCallback {
    LogMetadata log_metadata;
    LogMessageFn log_message_fn;
  };

  std::vector<StreamLogCallback> callbacks_;
};

/*************************Implementation Details******************************/

template <typename Request, typename Response>
absl::Status StreamLogger<Request, Response>::LogMessage() {
  absl::Status status;
  for (const auto& callback : callbacks_) {
    std::unique_ptr<google::protobuf::Message> log;
    absl::Status create_status = CreateLogMessage(callback.log_metadata, &log);
    if (create_status.ok()) {
      status.Update(callback.log_message_fn(*log));
    } else {
      LOG_EVERY_N_SEC(ERROR, 30)
          << "Failed creating log message for streaming request. Log metadata: "
          << callback.log_metadata.DebugString()
          << ", error: " << create_status;
      status.Update(create_status);
    }
  }
  return status;
}

template <typename Request, typename Response>
void StreamLogger<Request, Response>::AddLogCallback(
    const LogMetadata& log_metadata, LogMessageFn log_message_fn) {
  StreamLogCallback callback;
  callback.log_metadata = log_metadata;
  callback.log_message_fn = std::move(log_message_fn);
  callbacks_.push_back(std::move(callback));
}

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_CORE_STREAM_LOGGER_H_
