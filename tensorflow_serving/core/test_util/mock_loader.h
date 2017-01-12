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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_LOADER_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_LOADER_H_

#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {
namespace test_util {

class MockLoader : public Loader {
 public:
  MOCK_CONST_METHOD1(EstimateResources, Status(ResourceAllocation* estimate));
  MOCK_METHOD0(Load, Status());
  MOCK_METHOD0(Unload, void());
  MOCK_METHOD0(servable, AnyPtr());
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_MOCK_LOADER_H_
