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

// TensorFlow implementation of the RegressorInterface.

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_REGRESSOR_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_REGRESSOR_H_

#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/apis/regressor.h"

namespace tensorflow {
namespace serving {

// Create a new RegressorInterface backed by a TensorFlow SavedModel.
// Requires that the default SignatureDef be compatible with Regression.
Status CreateRegressorFromSavedModelBundle(
    const RunOptions& run_options, std::unique_ptr<SavedModelBundle> bundle,
    std::unique_ptr<RegressorInterface>* service);

// Create a new RegressorInterface backed by a TensorFlow Session using the
// specified SignatureDef. Does not take ownership of the Session.
// Useful in contexts where we need to avoid copying, e.g. if created per
// request. The caller must ensure that the session and signature live at least
// as long as the service.
Status CreateFlyweightTensorFlowRegressor(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    std::unique_ptr<RegressorInterface>* service);

// Similar to the above function, but with additional 'thread_pool_options'.
Status CreateFlyweightTensorFlowRegressor(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    const thread::ThreadPoolOptions& thread_pool_options,
    std::unique_ptr<RegressorInterface>* service);

// Get a regression signature from the meta_graph_def that's either:
// 1) The signature that model_spec explicitly specifies to use.
// 2) The default serving signature.
// If neither exist, or there were other issues, an error status is returned.
Status GetRegressionSignatureDef(const ModelSpec& model_spec,
                                 const MetaGraphDef& meta_graph_def,
                                 SignatureDef* signature);

// Validate a SignatureDef to make sure it's compatible with Regression.
// Populate the input and output tensor names, if the args are not nullptr.
//
// NOTE: output_tensor_names may already have elements in it (e.g. when building
// a full list of outputs from multiple signatures), and this function will just
// append to the vector.
Status PreProcessRegression(const SignatureDef& signature,
                            string* input_tensor_name,
                            std::vector<string>* output_tensor_names);

// Validate all results and populate a RegressionResult.
Status PostProcessRegressionResult(
    const SignatureDef& signature, int num_examples,
    const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, RegressionResult* result);

// Creates SavedModelTensorflowRegressor and runs Regression on it.
Status RunRegress(const RunOptions& run_options,
                  const MetaGraphDef& meta_graph_def,
                  const absl::optional<int64_t>& servable_version,
                  Session* session, const RegressionRequest& request,
                  RegressionResponse* response,
                  const thread::ThreadPoolOptions& thread_pool_options =
                      thread::ThreadPoolOptions());

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_REGRESSOR_H_
