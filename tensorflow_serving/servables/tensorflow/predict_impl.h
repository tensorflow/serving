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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// Utility methods for implementation of PredictionService::Predict.
class TensorflowPredictor {
 public:
  TensorflowPredictor() {}

  explicit TensorflowPredictor(ThreadPoolFactory* thread_pool_factory)
      : thread_pool_factory_(thread_pool_factory) {}

  Status Predict(const RunOptions& run_options, ServerCore* core,
                 const PredictRequest& request, PredictResponse* response);

  // Like Predict(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  Status PredictWithModelSpec(const RunOptions& run_options, ServerCore* core,
                              const ModelSpec& model_spec,
                              const PredictRequest& request,
                              PredictResponse* response);

 private:
  ThreadPoolFactory* thread_pool_factory_ = nullptr;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
