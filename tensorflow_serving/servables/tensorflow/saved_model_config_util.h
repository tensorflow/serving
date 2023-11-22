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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_UTIL_H_

#include <string>

#include "absl/status/statusor.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"

namespace tensorflow {
namespace serving {

// Name of the additional asset file containing a per model configuration proto.
inline constexpr char kSavedModelConfigPath[] = "saved_model_config.pb";

inline constexpr char kRemoteOpConfigRewriter[] = "remote_op_config_rewrite";
inline constexpr char kBatchOpRewriter[] = "batch_op_rewriter";

inline constexpr char kRemoteOpRewriteConfigParamKey[] =
    "remote_op_rewrite_config";
inline constexpr char kBatchOpRewriteConfigParamKey[] =
    "batch_op_rewrite_config";

// Extracts a `SavedModelConfig` proto from the optional asset file in the
// given directory. If the asset file does not exist, it returns an empty
// proto.
absl::StatusOr<SavedModelConfig> LoadSavedModelConfigOrDefault(
    const std::string& export_dir);

// Updates `rewrite_options` based on optimizers options in `session_overrides`.
void UpdateRewriterConfig(
    const tensorflow::serving::SessionOverrides& session_overrides,
    tensorflow::RewriterConfig* rewrite_options);

}  // namespace serving
}  // namespace tensorflow

#endif  //  #define
        //  TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_CONFIG_UTIL_H_
