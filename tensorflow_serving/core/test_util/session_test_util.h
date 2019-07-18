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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_

#include <functional>

#include "absl/base/attributes.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Sets a 'hook' function, which will be called when a new session is created
// via the tensorflow::NewSession() API. If the hook returns an error status,
// the session creation fails.
//
// For this hook to be enabled, create a session by setting
// SessionOptions::target as "new_session_hook/<actual_session_target>". This
// will call the hook as well as return the session created when target is
// "<actual_session_target>".
//
// Calling this method again replaces the previous hook.
//
// This method is NOT thread-safe.
ABSL_CONST_INIT extern const char kNewSessionHookSessionTargetPrefix[];
void SetNewSessionHook(std::function<Status(const SessionOptions&)> hook);

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_SESSION_TEST_UTIL_H_
