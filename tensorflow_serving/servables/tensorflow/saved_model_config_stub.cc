/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow_serving/servables/tensorflow/saved_model_config_stub.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"

namespace tensorflow {
namespace serving {
namespace {

class SavedModelConfigStubRegistry {
 public:
  SavedModelConfigStubRegistry()
      : stub_(std::make_unique<SavedModelConfigStub>()) {}

  void Register(std::unique_ptr<SavedModelConfigStub> stub) {
    stub_ = std::move(stub);
  }

  SavedModelConfigStub& Get() { return *stub_; }

 private:
  std::unique_ptr<SavedModelConfigStub> stub_;
};

SavedModelConfigStubRegistry& GetSavedModelConfigStubRegistry() {
  static auto* const registry = new SavedModelConfigStubRegistry;
  return *registry;
}

}  // namespace

void RegisterSavedModelConfigStub(std::unique_ptr<SavedModelConfigStub> stub) {
  GetSavedModelConfigStubRegistry().Register(std::move(stub));
}

Status ImportAndLoadSavedModelConfig(
    const std::string& export_dir, tensorflow::GraphOptions& graph_options,
    tensorflow::tfrt_stub::RuntimeConfig& runtime_config) {
  return GetSavedModelConfigStubRegistry().Get().ImportAndLoadSavedModelConfig(
      export_dir, graph_options, runtime_config);
}

}  // namespace serving
}  // namespace tensorflow
