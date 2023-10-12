/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TEST_UTIL_MOCK_TFRT_SAVED_MODEL
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TEST_UTIL_MOCK_TFRT_SAVED_MODEL

#include <gmock/gmock.h>
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"

namespace tensorflow {
namespace serving {
namespace test_util {

inline tfrt_stub::Runtime* GetTestTfrtRuntime() {
  static auto* const runtime =
      tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4).release();
  return runtime;
}

// A mock of tfrt::SavedModel.
class MockSavedModel : public tfrt::SavedModel {
 public:
  MockSavedModel() : SavedModel(GetTestTfrtRuntime()) {}

  MOCK_METHOD(const tensorflow::MetaGraphDef&, GetMetaGraphDef, (),
              (const, override));

  MOCK_METHOD(absl::optional<tfrt::FunctionMetadata>, GetFunctionMetadata,
              (absl::string_view func_name), (const, override));

  MOCK_METHOD(::tensorflow::Status, Run,
              (const tfrt::SavedModel::RunOptions& run_options,
               absl::string_view func_name, absl::Span<const Tensor> inputs,
               std::vector<Tensor>* outputs),
              (override));

  MOCK_METHOD(std::vector<std::string>, GetFunctionNames, (),
              (const, override));

  MOCK_METHOD(::tensorflow::Status, RunMultipleSignatures,
              (const tfrt::SavedModel::RunOptions& run_options,
               absl::Span<const std::string> names,
               absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
               std::vector<std::vector<tensorflow::Tensor>>* multi_outputs),
              (override));

  MOCK_METHOD(
      ::tensorflow::Status, RunByTensorNames,
      (const tfrt::SavedModel::RunOptions& run_options,
       (absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs),
       absl::Span<const std::string> output_tensor_names,
       absl::Span<const std::string> target_node_names,
       std::vector<tensorflow::Tensor>* outputs),
      (override));
};

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TEST_UTIL_MOCK_TFRT_SAVED_MODEL
