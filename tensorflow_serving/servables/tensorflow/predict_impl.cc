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

#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

Status TensorflowPredictor::Predict(const RunOptions& run_options,
                                    ServerCore* core,
                                    const PredictRequest& request,
                                    PredictResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  return PredictWithModelSpec(run_options, core, request.model_spec(), request,
                              response);
}

Status TensorflowPredictor::PredictWithModelSpec(const RunOptions& run_options,
                                                 ServerCore* core,
                                                 const ModelSpec& model_spec,
                                                 const PredictRequest& request,
                                                 PredictResponse* response) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  return internal::RunPredict(
      run_options, bundle->meta_graph_def, bundle.id().version,
      core->predict_response_tensor_serialization_option(),
      bundle->session.get(), request, response,
      thread_pool_factory_ == nullptr
          ? thread::ThreadPoolOptions()
          : thread_pool_factory_->GetThreadPools().get());
}

}  // namespace serving
}  // namespace tensorflow
