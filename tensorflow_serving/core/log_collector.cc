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

#include "tensorflow_serving/core/log_collector.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace serving {
namespace {

bool ParseLognamePrefix(StringPiece remaining, StringPiece* type) {
  return strings::Scanner(remaining)
      .OneLiteral("/")
      .RestartCapture()
      .ScanUntil('/')
      .GetResult(&remaining, type);
}

// This class is thread-safe.
class Registry {
 public:
  Status Register(const string& type, const LogCollector::Factory& factory)
      LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    const auto found_it = factory_map_.find(type);
    if (found_it != factory_map_.end()) {
      return errors::AlreadyExists("Type ", type, " already registered.");
    }
    factory_map_.insert({type, factory});
    return Status::OK();
  }

  const LogCollector::Factory* Lookup(const string& type) const
      LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    const auto found_it = factory_map_.find(type);
    if (found_it == factory_map_.end()) {
      return nullptr;
    }
    return &(found_it->second);
  }

 private:
  mutable mutex mu_;
  std::unordered_map<string, LogCollector::Factory> factory_map_
      GUARDED_BY(mu_);
};

auto* registry_ = new Registry();

}  // namespace

Status LogCollector::RegisterFactory(const string& type,
                                     const Factory& factory) {
  return registry_->Register(type, factory);
}

Status LogCollector::Create(
    const string& logname_prefix, const uint32 id,
    std::unique_ptr<LogCollector>* const log_collector) {
  StringPiece remaining(logname_prefix);
  StringPiece type;
  if (!ParseLognamePrefix(remaining, &type)) {
    return errors::InvalidArgument("Invalid logname_prefix: ", logname_prefix);
  }

  auto* factory = registry_->Lookup(type.ToString());
  if (factory == nullptr) {
    return errors::NotFound("Cannot find LogCollector::Factory for type: ",
                            type);
  }
  return (*factory)(remaining.ToString(), id, log_collector);
}

}  // namespace serving
}  // namespace tensorflow
