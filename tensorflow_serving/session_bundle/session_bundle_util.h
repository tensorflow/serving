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

#ifndef TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_UTIL_H_
#define TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_UTIL_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow_serving/session_bundle/manifest_proto.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"

namespace tensorflow {
namespace serving {

namespace session_bundle {

// Interface from bundle_shim.h

// Converts signatures in the MetaGraphDef into a SignatureDefs in the
// MetaGraphDef.
Status ConvertSignaturesToSignatureDefs(MetaGraphDef* meta_graph_def);

// Converts a SessionBundle to a SavedModelBundle.
Status ConvertSessionBundleToSavedModelBundle(
    SessionBundle& session_bundle, SavedModelBundle* saved_model_bundle);

// Loads a SavedModel from either a session-bundle path or a SavedModel bundle
// path. If `is_session_bundle` is not a nullptr, sets it to `true` iff
// SavedModel was up-converted and loaded from a SessionBundle.
// `is_session_bundle` value should not be used if error is returned.
Status LoadSessionBundleOrSavedModelBundle(
    const SessionOptions& session_options, const RunOptions& run_options,
    const string& export_dir, const std::unordered_set<string>& tags,
    SavedModelBundle* bundle, bool* is_session_bundle = nullptr);

// Interface from session_bundle.h

// Similar to the LoadSessionBundleFromPath(), but also allows the session run
// invocations for the restore and init ops to be configured with
// tensorflow::RunOptions.
//
// This method is EXPERIMENTAL and may change or be removed.
Status LoadSessionBundleFromPathUsingRunOptions(
    const SessionOptions& session_options, const RunOptions& run_options,
    const StringPiece export_dir, SessionBundle* bundle);

// Interface from signature.h

// (Re)set Signatures in a MetaGraphDef.
Status SetSignatures(const Signatures& signatures,
                     tensorflow::MetaGraphDef* meta_graph_def);

// Gets a ClassificationSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a ClassificationSignature,
// or does not exist.
Status GetClassificationSignature(
    const tensorflow::MetaGraphDef& meta_graph_def,
    ClassificationSignature* signature);

// Gets a RegressionSignature from a MetaGraphDef's default signature.
// Returns an error if the default signature is not a RegressionSignature,
// or does not exist.
Status GetRegressionSignature(const tensorflow::MetaGraphDef& meta_graph_def,
                              RegressionSignature* signature);

// Runs a classification using the provided signature and initialized Session.
//   input: input batch of items to classify
//   classes: output batch of classes; may be null if not needed
//   scores: output batch of scores; may be null if not needed
// Validates sizes of the inputs and outputs are consistent (e.g., input
// batch size equals output batch sizes).
// Does not do any type validation.
Status RunClassification(const ClassificationSignature& signature,
                         const Tensor& input, Session* session, Tensor* classes,
                         Tensor* scores);

// Runs regression using the provided signature and initialized Session.
//   input: input batch of items to run the regression model against
//   output: output targets
// Validates sizes of the inputs and outputs are consistent (e.g., input
// batch size equals output batch sizes).
// Does not do any type validation.
Status RunRegression(const RegressionSignature& signature, const Tensor& input,
                     Session* session, Tensor* output);

// Gets a named Signature from a MetaGraphDef.
// Returns an error if a Signature with the given name does not exist.
Status GetNamedSignature(const string& name,
                         const tensorflow::MetaGraphDef& meta_graph_def,
                         Signature* default_signature);

// EXPERIMENTAL. THIS METHOD MAY CHANGE OR GO AWAY. USE WITH CAUTION.
// Sets a global graph rewrite function that is called on all saved models
// immediately after metagraph load, but before session creation.  This function
// can only be called once.
Status SetGraphRewriter(
    std::function<Status(tensorflow::MetaGraphDef*)>&& rewriter);

}  // namespace session_bundle
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_UTIL_H_
