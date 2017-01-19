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

#ifndef TENSORFLOW_SERVING_CORE_LOG_COLLECTOR_H_
#define TENSORFLOW_SERVING_CORE_LOG_COLLECTOR_H_

#include <functional>
#include <memory>
#include <string>

#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/config/log_collector_config.pb.h"

namespace tensorflow {
namespace serving {

// LogCollector defines an abstract interface to use for collecting logs.
//
// Each LogCollector implementation is registered along with a 'type', and if a
// LogCollector is created using this API, we create the LogCollector
// corresponding to the 'type' specified.
class LogCollector {
 public:
  virtual ~LogCollector() = default;

  // Creates a log-collector for a given 'log_collector_config' and 'id'. The
  // factory registered for the type, mentioned in the config, can then be used
  // to create the log-collector. The 'id' argument helps in disambiguating logs
  // from replicated servers (processes), so it could be a combination of
  // task-id and replica-id or process-id and timestamp, etc.
  static Status Create(const LogCollectorConfig& log_collector_config,
                       const uint32 id,
                       std::unique_ptr<LogCollector>* log_collector);

  using Factory = std::function<decltype(Create)>;
  // Registers a factory for creating log-collectors for a particular 'type'.
  // Returns an error status if a factory is already registered for the
  // particular 'type'.
  static Status RegisterFactory(const string& type, const Factory& factory);

  // Collects the log as a protocol buffer.
  virtual Status CollectMessage(const google::protobuf::Message& message) = 0;

  // Flushes buffered data so that the data can survive an application crash
  // (but not an OS crash).
  virtual Status Flush() = 0;

 protected:
  LogCollector() = default;
};

namespace register_log_collector {

struct RegisterFactory {
  RegisterFactory(const string& type, const LogCollector::Factory& factory) {
    // This check happens during global object construction time, even before
    // control reaches main(), so we are ok with the crash.
    TF_CHECK_OK(LogCollector::RegisterFactory(type, factory));  // Crash ok.
  }
};

}  // namespace register_log_collector

}  // namespace serving
}  // namespace tensorflow

#define REGISTER_LOG_COLLECTOR_UNIQ_HELPER(ctr, type, factory) \
  REGISTER_LOG_COLLECTOR_UNIQ(ctr, type, factory)
#define REGISTER_LOG_COLLECTOR_UNIQ(ctr, type, factory)                   \
  static ::tensorflow::serving::register_log_collector::RegisterFactory   \
      register_lgc##ctr TF_ATTRIBUTE_UNUSED =                             \
          ::tensorflow::serving::register_log_collector::RegisterFactory( \
              type, factory)

// Registers a LogCollector factory implementation for a type.
#define REGISTER_LOG_COLLECTOR(type, factory) \
  REGISTER_LOG_COLLECTOR_UNIQ_HELPER(__COUNTER__, type, factory)

#endif  // TENSORFLOW_SERVING_CORE_LOG_COLLECTOR_H_
