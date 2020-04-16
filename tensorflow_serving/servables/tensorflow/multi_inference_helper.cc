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

#include "tensorflow_serving/servables/tensorflow/multi_inference_helper.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference.h"

namespace tensorflow {
namespace serving {

namespace {

const ModelSpec& GetModelSpecFromRequest(const MultiInferenceRequest& request) {
  if (request.tasks_size() > 0 && request.tasks(0).has_model_spec()) {
    return request.tasks(0).model_spec();
  }
  return ModelSpec::default_instance();
}

}  // namespace

Status RunMultiInferenceWithServerCore(
    const RunOptions& run_options, ServerCore* core,
    const tensorflow::thread::ThreadPoolOptions& thread_pool_options,
    const MultiInferenceRequest& request, MultiInferenceResponse* response) {
  return RunMultiInferenceWithServerCoreWithModelSpec(
      run_options, core, thread_pool_options, GetModelSpecFromRequest(request),
      request, response);
}

Status RunMultiInferenceWithServerCoreWithModelSpec(
    const RunOptions& run_options, ServerCore* core,
    const tensorflow::thread::ThreadPoolOptions& thread_pool_options,
    const ModelSpec& model_spec, const MultiInferenceRequest& request,
    MultiInferenceResponse* response) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));

  return RunMultiInference(run_options, bundle->meta_graph_def,
                           bundle.id().version, bundle->session.get(), request,
                           response, thread_pool_options);
}

}  // namespace serving
}  // namespace tensorflow
