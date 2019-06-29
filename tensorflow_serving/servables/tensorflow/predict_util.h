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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_UTIL_H_

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

namespace internal {
// Whether to serialize proto as field or content.
enum class PredictResponseTensorSerializationOption {
  kAsProtoField = 0,
  kAsProtoContent = 1,
};

// Similar to RunPredict below, but allows specification of a serialization
// option for the TensorProtos in the response.
Status RunPredict(
    const RunOptions& run_options, const MetaGraphDef& meta_graph_def,
    const optional<int64>& servable_version,
    const PredictResponseTensorSerializationOption tensor_serialization_option,
    Session* session, const PredictRequest& request, PredictResponse* response);

}  // namespace internal

// Implementation of Predict using the SavedModel SignatureDef format.
//
// IMPLEMENTATION NOTES: Calls the internal::RunPredict function above by
// specifying serialization option as kAsProtoField for backward compatibility.
Status RunPredict(const RunOptions& run_options,
                  const MetaGraphDef& meta_graph_def,
                  const optional<int64>& servable_version, Session* session,
                  const PredictRequest& request, PredictResponse* response);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_UTIL_H_
