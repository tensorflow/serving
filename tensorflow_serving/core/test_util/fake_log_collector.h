/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOG_COLLECTOR_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOG_COLLECTOR_H_

#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/log_collector.h"

namespace tensorflow {
namespace serving {

// FakeLogCollector which does nothing except count the number of times
// CollectMessage has been called on it.
class FakeLogCollector : public LogCollector {
 public:
  Status CollectMessage(const google::protobuf::Message& message) override {
    ++collect_count_;
    return Status::OK();
  }

  Status Flush() override { return Status::OK(); }

  int collect_count() const { return collect_count_; }

 private:
  int collect_count_ = 0;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_FAKE_LOG_COLLECTOR_H_
