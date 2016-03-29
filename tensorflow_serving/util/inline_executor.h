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

#ifndef TENSORFLOW_SERVING_UTIL_INLINE_EXECUTOR_H_
#define TENSORFLOW_SERVING_UTIL_INLINE_EXECUTOR_H_

#include <functional>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/util/executor.h"

namespace tensorflow {
namespace serving {

// An InlineExecutor is a trivial executor that immediately executes the closure
// given to it. It's useful as a mock, and in cases where an executor is needed,
// but multi-threadedness is not.
class InlineExecutor : public Executor {
 public:
  InlineExecutor();
  ~InlineExecutor() override;
  void Schedule(std::function<void()> fn) override;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_INLINE_EXECUTOR_H_
