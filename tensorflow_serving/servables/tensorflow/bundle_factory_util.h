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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_

#include "tensorflow/contrib/batching/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

// Returns SessionOptions based on the SessionBundleConfig.
// TODO(b/32248363): add SavedModelBundleConfig after we switch Model Server to
// Saved Model.
SessionOptions GetSessionOptions(const SessionBundleConfig& config);

// Returns RunOptions based on SessionBundleConfig.
// TODO(b/32248363): add SavedModelBundleConfig after we switch Model Server to
// Saved Model.
RunOptions GetRunOptions(const SessionBundleConfig& config);

// Creates a BatchScheduler based on the batching configuration.
Status CreateBatchScheduler(
    const BatchingParameters& batching_config,
    std::shared_ptr<SharedBatchScheduler<BatchingSessionTask>>*
        batch_scheduler);

// Estimates the resources a session bundle or saved model bundle will use once
// loaded, from its export or saved model path. tensorflow::Env::Default() will
// be used to access the file system.
//
// Uses the following crude heuristic, for now: estimated main-memory RAM =
// (combined size of all exported file(s)) * kResourceEstimateRAMMultiplier +
// kResourceEstimateRAMPadBytes.
// TODO(b/27694447): Improve the heuristic. At a minimum, account for GPU RAM.
Status EstimateResourceFromPath(const string& path,
                                ResourceAllocation* estimate);

// Similar to the above function, but also supplies a FileProbingEnv to use in
// lieu of tensorflow::Env::Default().
Status EstimateResourceFromPath(const string& path, FileProbingEnv* env,
                                ResourceAllocation* estimate);

// Wraps a session in a new session that automatically batches Run() calls, for
// the given signatures.
// TODO(b/33233998): Support batching for Run() calls that use a combination of
// signatures -- i.e. sometimes construct a single TensorSignature for a set of
// SignatureDefs (usually just two of them) -- based on some config.
Status WrapSessionForBatching(
    const BatchingParameters& batching_config,
    std::shared_ptr<SharedBatchScheduler<BatchingSessionTask>> batch_scheduler,
    const std::vector<SignatureDef>& signatures,
    std::unique_ptr<Session>* session);

// Wraps a session in a new session that only supports Run() without batching.
Status WrapSession(std::unique_ptr<Session>* session);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_BUNDLE_FACTORY_UTIL_H_
