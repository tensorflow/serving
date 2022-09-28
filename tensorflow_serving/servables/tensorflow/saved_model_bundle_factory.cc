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

#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_factory.h"

#include "absl/strings/string_view.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"
#include "tensorflow_serving/servables/tensorflow/tflite_session.h"
#include "tensorflow_serving/session_bundle/session_bundle_util.h"

namespace tensorflow {
namespace serving {

namespace {

// Extracts the signatures from 'bundle'.
std::vector<SignatureDef> GetSignatureDefs(const SavedModelBundle& bundle) {
  std::vector<SignatureDef> signature_defs;
  for (const auto& entry : bundle.meta_graph_def.signature_def()) {
    const SignatureDef& signature_def = entry.second;
    signature_defs.push_back(signature_def);
  }
  return signature_defs;
}

// TODO(b/140959776): Move this upstream alongside `kSavedModelFilenamePb`.
const char kTfLiteModelFilename[] = "model.tflite";

Status LoadTfLiteModel(const string& model_dir, SavedModelBundle* bundle,
                       const SessionOptions& options, int num_interpreter_pools,
                       int num_interpreters_per_pool) {
  std::unique_ptr<TfLiteSession> session;

  const string& fname = io::JoinPath(model_dir, kTfLiteModelFilename);
  uint64_t size;
  TF_RETURN_IF_ERROR(Env::Default()->GetFileSize(fname, &size));

  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(fname, &file));

  string model_bytes;
  model_bytes.resize(size);
  absl::string_view sv;
  TF_RETURN_IF_ERROR(file->Read(0, size, &sv, &model_bytes[0]));

  std::unique_ptr<TfLiteSession> tflite_session;
  TF_RETURN_IF_ERROR(TfLiteSession::Create(
      std::move(model_bytes), options, num_interpreter_pools,
      num_interpreters_per_pool, &tflite_session,
      bundle->meta_graph_def.mutable_signature_def()));
  bundle->session = std::move(tflite_session);
  return OkStatus();
}

bool TfLiteModelFound(const string& model_dir) {
  const string& fname = io::JoinPath(model_dir, kTfLiteModelFilename);
  return Env::Default()->FilesExist({fname}, nullptr);
}

}  // namespace

Status SavedModelBundleFactory::Create(
    const SessionBundleConfig& config,
    std::unique_ptr<SavedModelBundleFactory>* factory) {
  std::shared_ptr<Batcher> batcher;
  if (config.has_batching_parameters()) {
    TF_RETURN_IF_ERROR(
        CreateBatchScheduler(config.batching_parameters(), &batcher));
  }
  factory->reset(new SavedModelBundleFactory(config, batcher));
  return OkStatus();
}

Status SavedModelBundleFactory::EstimateResourceRequirement(
    const string& path, ResourceAllocation* estimate) const {
  return EstimateResourceFromPath(
      path, config_.resource_estimation_uses_validation_result(), estimate);
}

Status SavedModelBundleFactory::CreateSavedModelBundleWithMetadata(
    const Loader::Metadata& metadata, const string& path,
    std::unique_ptr<SavedModelBundle>* bundle) {
  return InternalCreateSavedModelBundle(metadata, path, bundle);
}

Status SavedModelBundleFactory::CreateSavedModelBundle(
    const string& path, std::unique_ptr<SavedModelBundle>* bundle) {
  return InternalCreateSavedModelBundle({}, path, bundle);
}

Status SavedModelBundleFactory::InternalCreateSavedModelBundle(
    const absl::optional<Loader::Metadata>& metadata, const string& path,
    std::unique_ptr<SavedModelBundle>* bundle) {
  bundle->reset(new SavedModelBundle);
  std::unordered_set<string> saved_model_tags(
      config_.saved_model_tags().begin(), config_.saved_model_tags().end());
  // Defaults to loading the meta graph def corresponding to the `serve` tag
  // if no `saved_model_tags` are specified.
  if (saved_model_tags.empty()) {
    saved_model_tags.insert(kSavedModelTagServe);
  }
  const auto& session_options = [&]() {
    auto result = GetSessionOptions(config_);
    if (metadata.has_value()) {
      auto* session_metadata =
          result.config.mutable_experimental()->mutable_session_metadata();
      session_metadata->set_name(metadata->servable_id.name);
      session_metadata->set_version(metadata->servable_id.version);
    }
    return result;
  }();

  bool is_tflite = config_.prefer_tflite_model() && TfLiteModelFound(path);
  if (is_tflite) {
    int num_tflite_pools = config_.num_tflite_pools();
    if (num_tflite_pools == 0 && config_.num_tflite_interpreters() > 0) {
      num_tflite_pools = config_.num_tflite_interpreters();
    }
    TF_RETURN_IF_ERROR(LoadTfLiteModel(
        path, bundle->get(), session_options, num_tflite_pools,
        config_.num_tflite_interpreters_per_pool()));
  } else {
    TF_RETURN_IF_ERROR(session_bundle::LoadSessionBundleOrSavedModelBundle(
        session_options, GetRunOptions(config_), path, saved_model_tags,
        bundle->get()));
  }
  if (config_.remove_unused_fields_from_bundle_metagraph()) {
    // Save memory by removing fields in MetaGraphDef proto message stored
    // in the bundle that we never use. Notably the unused graphdef submessage
    // can get large (MBs) wasting memory on the server.
    //
    // Presently we retain following field(s) of MetaGraphDef proto:
    // - signature_def
    MetaGraphDef metagraph;
    (*bundle)->meta_graph_def.Swap(&metagraph);
    (*bundle)->meta_graph_def.mutable_signature_def()->swap(
        *metagraph.mutable_signature_def());
  }
  if (config_.wrap_session_with_no_threading_params()) {
    return WrapSessionIgnoreThreadPoolOptions(&(*bundle)->session);
  } else if (config_.has_batching_parameters()) {
    absl::optional<BatchingParameters> batching_params;
    TF_RETURN_IF_ERROR(GetPerModelBatchingParams(
        path, config_.batching_parameters(),
        config_.enable_per_model_batching_params(), &batching_params));
    if (batching_params.has_value()) {
      // Enable batching of requests to any one signature_def in the SavedModel.
      // Note that in the future, the plan is to enable explicit configuration
      // of the one or many SignatureDefs to enable.
      const std::vector<SignatureDef> signatures = GetSignatureDefs(**bundle);
      return WrapSessionForBatching(batching_params.value(), batch_scheduler_,
                                    signatures, &(*bundle)->session);
    }
  }
  return WrapSession(&(*bundle)->session);
}

SavedModelBundleFactory::SavedModelBundleFactory(
    const SessionBundleConfig& config, std::shared_ptr<Batcher> batch_scheduler)
    : config_(config), batch_scheduler_(batch_scheduler) {}

}  // namespace serving
}  // namespace tensorflow
