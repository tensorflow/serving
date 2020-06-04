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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_H_

#include <string>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

// Run warmup requests to trigger lazy initializations (such as TF
// optimizations, XLA compilations) at load time, and consequently improve first
// request latency.
// Supported request types: Regress, Classify, Predict, MultiInference.
Status RunSavedModelWarmup(const ModelWarmupOptions& model_warmup_options,
                           const RunOptions& run_options,
                           const string& export_dir, SavedModelBundle* bundle);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_H_
