/* Copyright 2023 Google Inc. All Rights Reserved.

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
#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_STUB_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_STUB_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"

namespace tensorflow {
namespace serving {

// The tfrt native lowering stub that provides interface for internal and OSS
// with different impls.
class SavedModelConfigStub {
 public:
  virtual ~SavedModelConfigStub() = default;
  virtual Status ImportAndLoadSavedModelConfig(
      const std::string& export_dir, tensorflow::GraphOptions& graph_options,
      tensorflow::tfrt_stub::RuntimeConfig& runtime_config) {
    return absl::UnimplementedError("");
  }
};

void RegisterSavedModelConfigStub(std::unique_ptr<SavedModelConfigStub> stub);

Status ImportAndLoadSavedModelConfig(
    const std::string& export_dir, tensorflow::GraphOptions& graph_options,
    tensorflow::tfrt_stub::RuntimeConfig& runtime_config);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_STUB_H_
