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

// TensorFlow implementation of the RegressorInterface.

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSOR_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSOR_H_

#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/regressor.h"

namespace tensorflow {
namespace serving {
// Validate function's input and output.
Status PreProcessRegression(const tfrt::FunctionMetadata& function_metadata);

// Validate all results and populate a RegressionResult.
Status PostProcessRegressionResult(
    int num_examples, const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, RegressionResult* result);

// Run Regression.
Status RunRegress(const tfrt::SavedModel::RunOptions& run_options,
                  const absl::optional<int64_t>& servable_version,
                  tfrt::SavedModel* saved_model,
                  const RegressionRequest& request,
                  RegressionResponse* response);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSOR_H_
