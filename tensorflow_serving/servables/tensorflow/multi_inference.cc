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

#include "tensorflow_serving/servables/tensorflow/multi_inference.h"

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/servables/tensorflow/classifier.h"
#include "tensorflow_serving/servables/tensorflow/regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

Status TensorFlowMultiInferenceRunner::Infer(
    const RunOptions& run_options, const MultiInferenceRequest& request,
    MultiInferenceResponse* response) {
  TRACELITERAL("TensorFlowMultiInferenceRunner::Infer");

  string model_name = "";
  string input_tensor_name = "";
  std::set<string> signature_names;
  std::set<string> output_tensor_name_set;
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

    const string signature_name = task.model_spec().signature_name().empty()
                                      ? kDefaultServingSignatureDefKey
                                      : task.model_spec().signature_name();

    if (signature_names.find(signature_name) != signature_names.end()) {
      return errors::InvalidArgument(strings::StrCat(
          "Duplicate evaluation of signature: ", signature_name));
    }
    signature_names.insert(signature_name);

    auto iter = meta_graph_def_->signature_def().find(signature_name);
    if (iter == meta_graph_def_->signature_def().end()) {
      return errors::InvalidArgument(strings::StrCat(
          "Requested signature not found in model graph: ", signature_name));
    }
    string input_name;
    std::vector<string> output_names;

    if (task.method_name() == kClassifyMethodName) {
      TF_RETURN_IF_ERROR(
          PreProcessClassification(iter->second, &input_name, &output_names));
    } else if (task.method_name() == kRegressMethodName) {
      TF_RETURN_IF_ERROR(
          PreProcessRegression(iter->second, &input_name, &output_names));
    } else {
      return errors::Unimplemented("Unsupported signature method_name: ",
                                   task.method_name());
    }
    if (input_tensor_name.empty()) {
      input_tensor_name = input_name;
    } else if (input_tensor_name != input_name) {
      return errors::InvalidArgument(
          "Input tensor must be the same for all Signatures.");
    }

    for (const auto& output_tensor_name : output_names) {
      output_tensor_name_set.insert(output_tensor_name);
    }
  }

  const std::vector<string> output_tensor_names(output_tensor_name_set.begin(),
                                                output_tensor_name_set.end());

  std::vector<Tensor> outputs;
  int num_examples;
  TF_RETURN_IF_ERROR(PerformOneShotTensorComputation(
      run_options, request.input(), input_tensor_name, output_tensor_names,
      session_, &outputs, &num_examples));

  TRACELITERAL("PostProcessResults");
  for (const auto& task : request.tasks()) {
    const string signature_name = task.model_spec().signature_name().empty()
                                      ? kDefaultServingSignatureDefKey
                                      : task.model_spec().signature_name();
    auto iter = meta_graph_def_->signature_def().find(signature_name);
    if (iter == meta_graph_def_->signature_def().end()) {
      return errors::InvalidArgument(strings::StrCat(
          "Requested signature not found in model graph: ", signature_name));
    }
    if (task.method_name() == kClassifyMethodName) {
      TF_RETURN_IF_ERROR(PostProcessClassificationResult(
          iter->second, num_examples, output_tensor_names, outputs,
          response->add_results()->mutable_classification_result()));
    } else if (task.method_name() == kRegressMethodName) {
      TF_RETURN_IF_ERROR(PostProcessRegressionResult(
          iter->second, num_examples, output_tensor_names, outputs,
          response->add_results()->mutable_regression_result()));
    } else {
      return errors::InvalidArgument("Unrecognized signature method_name: ",
                                     task.method_name());
    }
  }
  return Status::OK();
}

namespace {

const ModelSpec& GetModelSpecFromRequest(const MultiInferenceRequest& request) {
  if (request.tasks_size() > 0 && request.tasks(0).has_model_spec()) {
    return request.tasks(0).model_spec();
  }
  return ModelSpec::default_instance();
}

}  // namespace

Status RunMultiInference(const RunOptions& run_options, ServerCore* core,
                         const MultiInferenceRequest& request,
                         MultiInferenceResponse* response) {
  TRACELITERAL("RunMultiInference");
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(
      core->GetServableHandle(GetModelSpecFromRequest(request), &bundle));

  TensorFlowMultiInferenceRunner inference_runner(bundle->session.get(),
                                                  &bundle->meta_graph_def);
  return inference_runner.Infer(run_options, request, response);
}

}  // namespace serving
}  // namespace tensorflow
