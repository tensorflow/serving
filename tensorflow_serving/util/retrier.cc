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

#include "tensorflow_serving/util/retrier.h"

#include <functional>

#include "absl/status/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace serving {

absl::Status Retry(const string& description, uint32 max_num_retries,
                   int64_t retry_interval_micros,
                   const std::function<absl::Status()>& retried_fn,
                   const std::function<bool(absl::Status)>& should_retry) {
  absl::Status status;
  int num_tries = 0;
  do {
    if (num_tries > 0) {
      Env::Default()->SleepForMicroseconds(retry_interval_micros);
      LOG(INFO) << "Retrying of " << description << " retry: " << num_tries;
    }
    status = retried_fn();
    if (!status.ok()) {
      LOG(ERROR) << description << " failed: " << status;
    }
    ++num_tries;
  } while (!status.ok() && num_tries < max_num_retries + 1 &&
           should_retry(status));

  if (!should_retry(status)) {
    LOG(INFO) << "Retrying of " << description << " was cancelled.";
  }
  if (num_tries == max_num_retries + 1) {
    LOG(INFO) << "Retrying of " << description
              << " exhausted max_num_retries: " << max_num_retries;
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
