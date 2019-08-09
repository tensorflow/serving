/* Copyright 2018 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/util/net_http/internal/net_logging.h"

#include <stddef.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "absl/base/attributes.h"
#include "absl/base/config.h"
#include "absl/base/log_severity.h"

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || \
    defined(__Fuchsia__) || defined(__native_client__) ||               \
    defined(__EMSCRIPTEN__)
#include <unistd.h>

#define NET_HAVE_POSIX_WRITE 1
#define NET_LOW_LEVEL_WRITE_SUPPORTED 1
#else
#undef NET_HAVE_POSIX_WRITE
#endif

#if (defined(__linux__) || defined(__FreeBSD__)) && !defined(__ANDROID__)
#include <sys/syscall.h>
#define NET_HAVE_SYSCALL_WRITE 1
#define NET_LOW_LEVEL_WRITE_SUPPORTED 1
#else
#undef NET_HAVE_SYSCALL_WRITE
#endif

#ifdef _WIN32
#include <io.h>

#define NET_HAVE_RAW_IO 1
#define NET_LOW_LEVEL_WRITE_SUPPORTED 1
#else
#undef NET_HAVE_RAW_IO
#endif

#ifdef NET_LOW_LEVEL_WRITE_SUPPORTED
static const char kTruncated[] = " ... (message truncated)\n";

inline static bool VADoNetLog(char** buf, int* size, const char* format,
                              va_list ap) ABSL_PRINTF_ATTRIBUTE(3, 0);
inline static bool VADoNetLog(char** buf, int* size, const char* format,
                              va_list ap) {
  int n = vsnprintf(*buf, *size, format, ap);
  bool result = true;
  if (n < 0 || n > *size) {
    result = false;
    if (static_cast<size_t>(*size) > sizeof(kTruncated)) {
      n = *size - sizeof(kTruncated);  // room for truncation message
    } else {
      n = 0;  // no room for truncation message
    }
  }
  *size -= n;
  *buf += n;
  return result;
}
#endif  // NET_LOW_LEVEL_WRITE_SUPPORTED

static constexpr int kLogBufSize = 10000;  // absl defaults to 3000

namespace {

bool DoNetLog(char** buf, int* size, const char* format, ...)
    ABSL_PRINTF_ATTRIBUTE(3, 4);
bool DoNetLog(char** buf, int* size, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  int n = vsnprintf(*buf, *size, format, ap);
  va_end(ap);
  if (n < 0 || n > *size) return false;
  *size -= n;
  *buf += n;
  return true;
}

void NetLogVA(absl::LogSeverity severity, const char* file, int line,
              const char* format, va_list ap) ABSL_PRINTF_ATTRIBUTE(4, 0);
void NetLogVA(absl::LogSeverity severity, const char* file, int line,
              const char* format, va_list ap) {
  char buffer[kLogBufSize];
  char* buf = buffer;
  int size = sizeof(buffer);
#ifdef NET_LOW_LEVEL_WRITE_SUPPORTED
  bool enabled = true;
#else
  bool enabled = false;
#endif

#ifdef ABSL_MIN_LOG_LEVEL
  if (severity < static_cast<absl::LogSeverity>(ABSL_MIN_LOG_LEVEL) &&
      severity < absl::LogSeverity::kFatal) {
    enabled = false;
  }
#endif

  if (enabled) {
    DoNetLog(&buf, &size, "[%s : %d] NET_LOG: ", file, line);
  }

#ifdef NET_LOW_LEVEL_WRITE_SUPPORTED
  if (enabled) {
    bool no_chop = VADoNetLog(&buf, &size, format, ap);
    if (no_chop) {
      DoNetLog(&buf, &size, "\n");
    } else {
      DoNetLog(&buf, &size, "%s", kTruncated);
    }
    tensorflow::serving::net_http::SafeWriteToStderr(buffer, strlen(buffer));
  }
#else
  static_cast<void>(format);
  static_cast<void>(ap);
#endif

  if (severity == absl::LogSeverity::kFatal) {
    abort();
  }
}

}  // namespace

namespace tensorflow {
namespace serving {
namespace net_http {

void SafeWriteToStderr(const char* s, size_t len) {
#if defined(NET_HAVE_SYSCALL_WRITE)
  syscall(SYS_write, STDERR_FILENO, s, len);
#elif defined(NET_HAVE_POSIX_WRITE)
  write(STDERR_FILENO, s, len);
#elif defined(NET_HAVE_RAW_IO)
  _write(/* stderr */ 2, s, len);
#else
  (void)s;
  (void)len;
#endif
}

void NetLog(absl::LogSeverity severity, const char* file, int line,
            const char* format, ...) ABSL_PRINTF_ATTRIBUTE(4, 5);
void NetLog(absl::LogSeverity severity, const char* file, int line,
            const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  NetLogVA(severity, file, line, format, ap);
  va_end(ap);
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
