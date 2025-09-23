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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"

namespace tensorflow {
namespace serving {

// Returns error if the `assets.extra/saved_model_config.pb` cannot be parsed.
// Returns success otherwise (including empty or no `saved_model_config.pb`).
// On success, reads SavedModelConfig proto from the specified model directory,
// adds or replaces some optimization options in
// `tensorflow::serving::RewriterConfig` of `tensorflow::GraphOptions` and
// replaces the `runtime_config`.
Status LoadSavedModelConfig(
    const std::string& export_dir, tensorflow::GraphOptions& graph_options,
    tensorflow::tfrt_stub::RuntimeConfig& runtime_config);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_H_
