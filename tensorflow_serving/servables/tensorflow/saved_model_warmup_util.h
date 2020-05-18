/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_UTIL_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace internal {

struct WarmupConsts {
  static constexpr char kRequestsFileName[] = "tf_serving_warmup_requests";
  static constexpr int kMaxNumRecords = 1000;
};

// Reads sample warmup requests from assets.extra/tf_serving_warmup_requests
// file (if exists) and invokes them one by one on the given saved_model,
// to trigger lazy initializations (such as TF optimizations, XLA compilations)
// at load time, and consequently improve first request latency.
// Warmup is skipped if no warmup file present.
Status RunSavedModelWarmup(
    const ModelWarmupOptions& model_warmup_options, const string export_dir,
    std::function<Status(PredictionLog)> warmup_request_executor);

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_UTIL_H_
