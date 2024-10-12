/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/session_bundle/session_bundle_util.h"

#include <functional>
#include <unordered_set>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow_serving/session_bundle/manifest_proto.h"

namespace tensorflow {
namespace serving {

namespace session_bundle {

absl::Status ConvertSignaturesToSignatureDefs(MetaGraphDef* meta_graph_def) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status ConvertSessionBundleToSavedModelBundle(
    SessionBundle& session_bundle, SavedModelBundle* saved_model_bundle) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status LoadSessionBundleOrSavedModelBundle(
    const SessionOptions& session_options, const RunOptions& run_options,
    const string& export_dir, const std::unordered_set<string>& tags,
    bool maybe_load_saved_model_config, SavedModelBundle* bundle,
    bool* is_session_bundle) {
  if (maybe_load_saved_model_config) {
    return errors::Unimplemented(
        "Saved model config functionality is not implemented.");
  }
  return LoadSessionBundleOrSavedModelBundle(session_options, run_options,
                                             export_dir, tags, bundle,
                                             is_session_bundle);
}

absl::Status LoadSessionBundleOrSavedModelBundle(
    const SessionOptions& session_options, const RunOptions& run_options,
    const string& export_dir, const std::unordered_set<string>& tags,
    SavedModelBundle* bundle, bool* is_session_bundle) {
  if (is_session_bundle != nullptr) {
    *is_session_bundle = false;
  }
  if (Env::Default()
          ->FileExists(io::JoinPath(export_dir, "export.meta"))
          .ok()) {
    return errors::Unimplemented("Session Bundle is deprecated and removed.");
  }
  if (MaybeSavedModelDirectory(export_dir)) {
    return LoadSavedModel(session_options, run_options, export_dir, tags,
                          bundle);
  }
  return absl::Status(
      static_cast<absl::StatusCode>(absl::StatusCode::kNotFound),
      strings::StrCat("Specified file path does not appear to contain a "
                      "SavedModel bundle (should have a file called "
                      "`saved_model.pb`)\n"
                      "Specified file path: ",
                      export_dir));
}

absl::Status LoadSessionBundleFromPathUsingRunOptions(
    const SessionOptions& session_options, const RunOptions& run_options,
    const StringPiece export_dir, SessionBundle* bundle) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status SetSignatures(const Signatures& signatures,
                           tensorflow::MetaGraphDef* meta_graph_def) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status GetClassificationSignature(
    const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status GetRegressionSignature(
    const tensorflow::MetaGraphDef& meta_graph_def,
    RegressionSignature* signature) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status RunClassification(const ClassificationSignature& signature,
                               const Tensor& input, Session* session,
                               Tensor* classes, Tensor* scores) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status RunRegression(const RegressionSignature& signature,
                           const Tensor& input, Session* session,
                           Tensor* output) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

absl::Status GetNamedSignature(const string& name,
                               const tensorflow::MetaGraphDef& meta_graph_def,
                               Signature* default_signature) {
  return errors::Unimplemented("Session Bundle is deprecated and removed.");
}

}  // namespace session_bundle
}  // namespace serving
}  // namespace tensorflow
