/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_HELPER_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_HELPER_H_

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// Runs MultiInference
Status RunMultiInferenceWithServerCore(const RunOptions& run_options,
                                       ServerCore* core,
                                       const MultiInferenceRequest& request,
                                       MultiInferenceResponse* response);

// Like RunMultiInferenceWithServerCore(), but uses 'model_spec' instead of the
// one(s) embedded in 'request'.
Status RunMultiInferenceWithServerCoreWithModelSpec(
    const RunOptions& run_options, ServerCore* core,
    const ModelSpec& model_spec, const MultiInferenceRequest& request,
    MultiInferenceResponse* response);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_HELPER_H_
