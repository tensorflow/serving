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

#include "tensorflow_serving/core/test_util/session_test_util.h"

#include "absl/memory/memory.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {
namespace test_util {
namespace {

using NewSessionHook = std::function<Status(const SessionOptions&)>;
NewSessionHook new_session_hook_;

NewSessionHook GetNewSessionHook() { return new_session_hook_; }

// A DelegatingSessionFactory is used to setup the new-session-hook.
//
// This SessionFactory accepts "new_session_hook/<actual_session_target>" as the
// session target. While returning the created session, it calls the
// new-session-hook and returns the session created when target is
// "<actual_session_target>".
class DelegatingSessionFactory : public SessionFactory {
 public:
  DelegatingSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return absl::StartsWith(options.target, "new_session_hook/");
  }

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    auto actual_session_options = options;
    actual_session_options.target = std::string(
        absl::StripPrefix(options.target, kNewSessionHookSessionTargetPrefix));
    auto new_session_hook = GetNewSessionHook();
    if (new_session_hook) {
      TF_RETURN_IF_ERROR(new_session_hook(actual_session_options));
    }
    Session* actual_session;
    TF_RETURN_IF_ERROR(
        tensorflow::NewSession(actual_session_options, &actual_session));
    *out_session = actual_session;
    return Status::OK();
  }
};

class DelegatingSessionRegistrar {
 public:
  DelegatingSessionRegistrar() {
    SessionFactory::Register("DELEGATING_SESSION",
                             new DelegatingSessionFactory());
  }
};
static DelegatingSessionRegistrar registrar;

}  // namespace

const char kNewSessionHookSessionTargetPrefix[] = "new_session_hook/";

void SetNewSessionHook(NewSessionHook hook) {
  new_session_hook_ = std::move(hook);
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
