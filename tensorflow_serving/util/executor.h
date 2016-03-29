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

#ifndef TENSORFLOW_SERVING_UTIL_EXECUTOR_H_
#define TENSORFLOW_SERVING_UTIL_EXECUTOR_H_

#include <functional>

namespace tensorflow {
namespace serving {

// An abstract object that can execute closures.
//
// Implementations of executor must be thread-safe.
class Executor {
 public:
  virtual ~Executor() = default;

  // Schedule the specified 'fn' for execution in this executor. Depending on
  // the subclass implementation, this may block in some situations.
  virtual void Schedule(std::function<void()> fn) = 0;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_EXECUTOR_H_
