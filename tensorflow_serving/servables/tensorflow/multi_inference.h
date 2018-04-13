/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_H_

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// TensorFlow implementation of the MultiInference.
// Only supports Models in the SavedModel format.
class TensorFlowMultiInferenceRunner {
 public:
  TensorFlowMultiInferenceRunner(Session* session,
                                 const MetaGraphDef* meta_graph_def)
      : TensorFlowMultiInferenceRunner(session, meta_graph_def,
                                       /*servable_version*/ {}) {}

  TensorFlowMultiInferenceRunner(Session* session,
                                 const MetaGraphDef* meta_graph_def,
                                 optional<int64> servable_version)
      : session_(session),
        meta_graph_def_(meta_graph_def),
        servable_version_(servable_version) {}

  // Run inference and return the inference results in the same order as the
  // InferenceTasks in the request.
  Status Infer(const RunOptions& run_options,
               const MultiInferenceRequest& request,
               MultiInferenceResponse* response);

  virtual ~TensorFlowMultiInferenceRunner() = default;

 private:
  Session* const session_;
  const MetaGraphDef* const meta_graph_def_;
  // If available, servable_version is used to set the ModelSpec version in the
  // InferenceResults of the MultiInferenceResponse.
  const optional<int64> servable_version_;
};

// Creates TensorFlowMultiInferenceRunner and calls Infer on it.
Status RunMultiInference(const RunOptions& run_options,
                         const MetaGraphDef& meta_graph_def,
                         const optional<int64>& servable_version,
                         Session* session, const MultiInferenceRequest& request,
                         MultiInferenceResponse* response);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MULTI_INFERENCE_H_
