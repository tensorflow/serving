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
#include "tensorflow_serving/servables/tensorflow/saved_model_config.h"

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"

namespace tensorflow {
namespace serving {

absl::Status LoadSavedModelConfig(
    const std::string& export_dir, tensorflow::GraphOptions& graph_options,
    tensorflow::tfrt_stub::RuntimeConfig& runtime_config) {
  absl::StatusOr<SavedModelConfig> model_config =
      LoadSavedModelConfigOrDefault(export_dir);
  if (!model_config.ok()) {
    return model_config.status();
  }

  if (model_config->has_session_overrides()) {
    UpdateRewriterConfig(model_config->session_overrides(),
                         graph_options.mutable_rewrite_options());
  }

  if (model_config->has_tfrt_runtime_config()) {
    auto created_runtime_config =
        tensorflow::tfrt_stub::RuntimeConfig::CreateFromProto(
            model_config->tfrt_runtime_config());
    if (created_runtime_config.ok()) {
      runtime_config = std::move(*created_runtime_config);
    } else {
      return created_runtime_config.status();
    }
  }

  return absl::Status();
}

}  // namespace serving
}  // namespace tensorflow
