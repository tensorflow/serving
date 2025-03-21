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

#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/types.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/stringpiece.h"
#include "tensorflow_serving/servables/tensorflow/remote_op_config_rewriter.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"

namespace tensorflow {
namespace serving {
namespace {
void AddOrReplaceOptimizer(const std::string& custom_optimizer_name,
                           const std::string& parameter_key,
                           const std::string& parameter_value,
                           RewriterConfig* rewrite_options) {
  google::protobuf::Map<std::string, AttrValue>* parameter_map = nullptr;
  for (auto& custom_optimizer : *rewrite_options->mutable_custom_optimizers()) {
    if (custom_optimizer.name() == custom_optimizer_name) {
      parameter_map = custom_optimizer.mutable_parameter_map();
      break;
    }
  }

  if (parameter_map == nullptr) {
    auto* custom_optimizer = rewrite_options->add_custom_optimizers();
    custom_optimizer->set_name(custom_optimizer_name);
    parameter_map = custom_optimizer->mutable_parameter_map();
  }

  (*parameter_map)[parameter_key].set_s(absl::Base64Escape(parameter_value));
}
}  // namespace

void UpdateRewriterConfig(
    const tensorflow::serving::SessionOverrides& session_overrides,
    tensorflow::RewriterConfig* rewrite_options) {
  DCHECK(rewrite_options != nullptr);

  // Grappler options for RemoteOpConfigRewriter.
  if (session_overrides.has_remote_op_remap_config()) {
    AddOrReplaceOptimizer(
        kRemoteOpConfigRewriter, kRemoteOpRewriteConfigParamKey,
        session_overrides.remote_op_remap_config().SerializeAsString(),
        rewrite_options);
  }

  // Grappler options for BatchOpRewriter.
  if (session_overrides.has_batch_op_rewriter_config()) {
    AddOrReplaceOptimizer(
        kBatchOpRewriter, kBatchOpRewriteConfigParamKey,
        session_overrides.batch_op_rewriter_config().SerializeAsString(),
        rewrite_options);
  }

  // Other rewriter options.
  rewrite_options->set_disable_meta_optimizer(
      session_overrides.disable_meta_optimizer());
}

absl::StatusOr<SavedModelConfig> LoadSavedModelConfigOrDefault(
    const std::string& export_dir) {
  const std::string saved_model_config_path = tsl::io::JoinPath(
      export_dir, kSavedModelAssetsExtraDirectory, kSavedModelConfigPath);
  SavedModelConfig saved_model_config;
  if (!tsl::Env::Default()->FilesExist({saved_model_config_path}, nullptr)) {
    // SavedModelConfig file is optional and may not exist.
    return saved_model_config;
  }

  LOG(INFO) << "Loading model config from " << saved_model_config_path;
  std::string content;
  tsl::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSize(saved_model_config_path, &file_size));
  content.resize(file_size);

  std::unique_ptr<tsl::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->NewRandomAccessFile(saved_model_config_path, &file));

  absl::string_view result;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &result, &(content)[0]));

  if (!saved_model_config.ParseFromString(content)) {
    return tsl::errors::Internal("Unable to parse SavedModelConfig: ",
                                 saved_model_config_path);
  }
  LOG(INFO) << "Finished loading model config from " << saved_model_config_path
            << ":" << saved_model_config.DebugString();
  return saved_model_config;
}

}  // namespace serving
}  // namespace tensorflow
