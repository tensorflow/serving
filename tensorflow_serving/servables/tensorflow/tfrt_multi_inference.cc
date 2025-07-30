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

#include "tensorflow_serving/servables/tensorflow/tfrt_multi_inference.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/error_logging.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_classifier.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

absl::Status RunMultiInference(const tfrt::SavedModel::RunOptions& run_options,
                               const absl::optional<int64_t>& servable_version,
                               tfrt::SavedModel* saved_model,
                               const MultiInferenceRequest& request,
                               MultiInferenceResponse* response) {
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(
      InputToSerializedExampleTensor(request.input(), &input_tensor));
  std::vector<std::vector<Tensor>> input_tensors;
  int num_examples = input_tensor.dim_size(0);
  input_tensors.resize(request.tasks_size());
  for (int i = 0; i < request.tasks_size(); ++i) {
    input_tensors[i].emplace_back(input_tensor);
  }

  // Pre-processing.
  std::string model_name = "";
  std::set<std::string> function_names_set;
  std::vector<std::string> function_names;
  function_names.reserve(request.tasks_size());
  for (const auto& task : request.tasks()) {
    if (task.model_spec().name().empty()) {
      return errors::InvalidArgument(
          "Found ModelSpec with an empty model name.");
    }
    if (model_name.empty()) {
      model_name = task.model_spec().name();
    } else if (model_name != task.model_spec().name()) {
      return errors::InvalidArgument(
          "All ModelSpecs in a MultiInferenceRequest must access the same "
          "model name.");
    }

    const std::string function_name = task.model_spec().signature_name().empty()
                                          ? kDefaultServingSignatureDefKey
                                          : task.model_spec().signature_name();

    // TODO(b/183949363): Remove the constrain here. We could allow duplicated
    // function names and simply return result for each of them.
    if (function_names_set.find(function_name) != function_names_set.end()) {
      return errors::InvalidArgument(strings::StrCat(
          "Duplicate evaluation of signature: ", function_name));
    }
    function_names_set.insert(function_name);
    function_names.push_back(function_name);

    const auto function_metadata =
        saved_model->GetFunctionMetadata(function_name);
    if (!function_metadata.has_value()) {
      return errors::InvalidArgument(
          strings::StrCat("Function \"", function_name, "\" not found."));
    }

    if (task.method_name() == kClassifyMethodName) {
      TF_RETURN_IF_ERROR(PreProcessClassification(function_metadata.value()));
    } else if (task.method_name() == kRegressMethodName) {
      TF_RETURN_IF_ERROR(PreProcessRegression(function_metadata.value()));
    } else {
      return errors::Unimplemented("Unsupported signature method_name: ",
                                   task.method_name());
    }
  }

  // Executes requests.
  std::vector<std::vector<Tensor>> output_tensors;
  if (const auto status = saved_model->RunMultipleSignatures(
          run_options, function_names, input_tensors, &output_tensors);
      !status.ok()) {
    if (IsTfrtErrorLoggingEnabled()) {
      tsl::error_logging::Log("TFRT", "SavedModelRun", status.message())
          .IgnoreError();
    }
    return status;
  }

  // Post-processing.
  for (int i = 0; i < request.tasks_size(); ++i) {
    // We have already checked the existence of the function metadata before
    // execution.
    const auto function_metadata =
        saved_model->GetFunctionMetadata(function_names[i]);
    DCHECK(function_metadata.has_value());
    if (request.tasks(i).method_name() == kClassifyMethodName) {
      TF_RETURN_IF_ERROR(PostProcessClassificationResult(
          num_examples, function_metadata->GetOutputNames(), output_tensors[i],
          response->add_results()->mutable_classification_result()));
    } else if (request.tasks(i).method_name() == kRegressMethodName) {
      TF_RETURN_IF_ERROR(PostProcessRegressionResult(
          num_examples, function_metadata->GetOutputNames(), output_tensors[i],
          response->add_results()->mutable_regression_result()));
    } else {
      return errors::InvalidArgument("Unrecognized signature method_name: ",
                                     request.tasks(i).method_name());
    }
    MakeModelSpec(request.tasks(i).model_spec().name(),
                  request.tasks(i).model_spec().signature_name(),
                  servable_version,
                  response->mutable_results(response->results_size() - 1)
                      ->mutable_model_spec());
  }
  RecordRequestExampleCount(model_name, num_examples);
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
