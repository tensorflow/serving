/* Copyright 2019 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_INTERNAL_NET_LOGGING_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_INTERNAL_NET_LOGGING_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/base/log_severity.h"
#include "absl/base/macros.h"
#include "absl/base/port.h"

// This initial version is a minimum fork from absl/base/internal/raw_logging.h
// Hooks are not supported.
//
// TODO(wenboz): finalize the log support for net_http
// * make logging pluggable (by TF serving or by adapting external libraries

#define NET_LOG(severity, ...)                                               \
  do {                                                                       \
    constexpr const char* net_logging_internal_basename =                    \
        ::tensorflow::serving::net_http::Basename(__FILE__,                  \
                                                  sizeof(__FILE__) - 1);     \
    ::tensorflow::serving::net_http::NetLog(NET_LOGGING_INTERNAL_##severity, \
                                            net_logging_internal_basename,   \
                                            __LINE__, __VA_ARGS__);          \
  } while (0)

#define NET_CHECK(condition, message)                             \
  do {                                                            \
    if (ABSL_PREDICT_FALSE(!(condition))) {                       \
      NET_LOG(FATAL, "Check %s failed: %s", #condition, message); \
    }                                                             \
  } while (0)

#define NET_LOGGING_INTERNAL_INFO ::absl::LogSeverity::kInfo
#define NET_LOGGING_INTERNAL_WARNING ::absl::LogSeverity::kWarning
#define NET_LOGGING_INTERNAL_ERROR ::absl::LogSeverity::kError
#define NET_LOGGING_INTERNAL_FATAL ::absl::LogSeverity::kFatal
#define NET_LOGGING_INTERNAL_LEVEL(severity) \
  ::absl::NormalizeLogSeverity(severity)

namespace tensorflow {
namespace serving {
namespace net_http {

void NetLog(absl::LogSeverity severity, const char* file, int line,
            const char* format, ...) ABSL_PRINTF_ATTRIBUTE(4, 5);

void SafeWriteToStderr(const char* s, size_t len);

constexpr const char* Basename(const char* fname, int offset) {
  return offset == 0 || fname[offset - 1] == '/' || fname[offset - 1] == '\\'
             ? fname + offset
             : Basename(fname, offset - 1);
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_INTERNAL_NET_LOGGING_H_
