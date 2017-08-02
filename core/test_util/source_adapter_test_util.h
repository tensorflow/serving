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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_SOURCE_ADAPTER_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_SOURCE_ADAPTER_TEST_UTIL_H_

#include <algorithm>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {
template <typename InputType, typename OutputType>
class SourceAdapter;
}  // namespace serving
}  // namespace tensorflow

namespace tensorflow {
namespace serving {
namespace test_util {

// Takes a SourceAdapter, and arranges to push a single data item of type
// InputType through its aspired-versions API. Returns the resulting
// ServableData<OutputType> object (which has a trivial servable id of {"", 0}).
//
// Assumes 'adapter->SetAspiredVersionsCallback()' has not yet been called.
// Assumes 'adapter->SetAspiredVersions()' is synchronous and internally calls
// the outgoing aspired-versions callback.
//
// Mutates 'adapter' and leaves it in an unspecified state.
template <typename InputType, typename OutputType>
ServableData<OutputType> RunSourceAdapter(
    const InputType& in, SourceAdapter<InputType, OutputType>* adapter);

//////////
// Implementation details follow. API users need not read.

template <typename InputType, typename OutputType>
ServableData<OutputType> RunSourceAdapter(
    const InputType& in, SourceAdapter<InputType, OutputType>* adapter) {
  ServableId servable_id = {"", 0};
  std::vector<ServableData<OutputType>> servable_data;
  bool outgoing_callback_called = false;
  adapter->SetAspiredVersionsCallback(
      [&servable_id, &servable_data, &outgoing_callback_called](
          const StringPiece servable_name,
          std::vector<ServableData<OutputType>> versions) {
        outgoing_callback_called = true;
        CHECK_EQ(servable_id.name, servable_name);
        servable_data = std::move(versions);
      });
  adapter->SetAspiredVersions(servable_id.name,
                              {CreateServableData(servable_id, in)});
  CHECK(outgoing_callback_called)
      << "Supplied adapter appears to have asynchronous behavior";
  CHECK_EQ(1, servable_data.size());
  CHECK_EQ(servable_id, servable_data[0].id());
  return std::move(servable_data[0]);
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_SOURCE_ADAPTER_TEST_UTIL_H_
