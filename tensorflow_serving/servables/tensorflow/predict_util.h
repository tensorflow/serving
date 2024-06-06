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

#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"

namespace tensorflow {
namespace serving {

namespace internal {

// Similar to RunPredict below, but allows specification of a serialization
// option for the TensorProtos in the response.
Status RunPredict(
    const RunOptions& run_options, const MetaGraphDef& meta_graph_def,
    const absl::optional<int64_t>& servable_version,
    const PredictResponseTensorSerializationOption tensor_serialization_option,
    Session* session, const PredictRequest& request, PredictResponse* response,
    const thread::ThreadPoolOptions& thread_pool_options =
        thread::ThreadPoolOptions());

// Validate a SignatureDef to make sure it's compatible with prediction, and
// if so, populate the input and output tensor names.
Status PreProcessPrediction(const SignatureDef& signature,
                            const PredictRequest& request,
                            std::vector<std::pair<string, Tensor>>* inputs,
                            std::vector<string>* output_tensor_names,
                            std::vector<string>* output_tensor_aliases);

// Validate results and populate a PredictResponse.
// Tensors are serialized as specified.
Status PostProcessPredictionResult(
    const std::vector<string>& output_tensor_aliases,
    const std::vector<Tensor>& output_tensors,
    const internal::PredictResponseTensorSerializationOption option,
    PredictResponse* response);

}  // namespace internal

// Implementation of Predict using the SavedModel SignatureDef format.
//
// IMPLEMENTATION NOTES: Calls the internal::RunPredict function above by
// specifying serialization option as kAsProtoField for backward compatibility.
Status RunPredict(const RunOptions& run_options,
                  const MetaGraphDef& meta_graph_def,
                  const absl::optional<int64_t>& servable_version,
                  Session* session, const PredictRequest& request,
                  PredictResponse* response,
                  const thread::ThreadPoolOptions& thread_pool_options =
                      thread::ThreadPoolOptions());

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_UTIL_H_
