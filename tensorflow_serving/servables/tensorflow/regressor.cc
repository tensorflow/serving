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

#include "tensorflow_serving/servables/tensorflow/regressor.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/apis/regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

// Implementation of the RegressorInterface using SavedModel.
class SavedModelTensorFlowRegressor : public RegressorInterface {
 public:
  explicit SavedModelTensorFlowRegressor(
      const RunOptions& run_options, Session* session,
      const SignatureDef* const signature,
      const thread::ThreadPoolOptions& thread_pool_options =
          thread::ThreadPoolOptions())
      : run_options_(run_options),
        session_(session),
        signature_(signature),
        thread_pool_options_(thread_pool_options) {}

  ~SavedModelTensorFlowRegressor() override = default;

  Status Regress(const RegressionRequest& request,
                 RegressionResult* result) override {
    TRACELITERAL("SavedModelTensorFlowRegressor::Regress");

    string input_tensor_name;
    std::vector<string> output_tensor_names;
    TF_RETURN_IF_ERROR(PreProcessRegression(*signature_, &input_tensor_name,
                                            &output_tensor_names));

    std::vector<Tensor> outputs;
    int num_examples;
    int64_t runtime_latency;
    TF_RETURN_IF_ERROR(PerformOneShotTensorComputation(
        run_options_, request.input(), input_tensor_name, output_tensor_names,
        session_, &outputs, &num_examples, thread_pool_options_,
        &runtime_latency));
    RecordRuntimeLatency(request.model_spec().name(), /*api=*/"Regress",
                         /*runtime=*/"TF1", runtime_latency);

    TRACELITERAL("ConvertToRegressionResult");
    return PostProcessRegressionResult(*signature_, num_examples,
                                       output_tensor_names, outputs, result);
  }

 private:
  const RunOptions run_options_;
  Session* const session_;
  const SignatureDef* const signature_;
  const thread::ThreadPoolOptions thread_pool_options_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelTensorFlowRegressor);
};

class SavedModelRegressor : public RegressorInterface {
 public:
  SavedModelRegressor(const RunOptions& run_options,
                      std::unique_ptr<SavedModelBundle> bundle)
      : run_options_(run_options), bundle_(std::move(bundle)) {}

  ~SavedModelRegressor() override = default;

  Status Regress(const RegressionRequest& request,
                 RegressionResult* result) override {
    SignatureDef signature;
    TF_RETURN_IF_ERROR(GetRegressionSignatureDef(
        request.model_spec(), bundle_->meta_graph_def, &signature));
    SavedModelTensorFlowRegressor regressor(run_options_,
                                            bundle_->session.get(), &signature);
    return regressor.Regress(request, result);
  }

 private:
  const RunOptions run_options_;
  std::unique_ptr<SavedModelBundle> bundle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelRegressor);
};

}  // namespace

Status CreateRegressorFromSavedModelBundle(
    const RunOptions& run_options, std::unique_ptr<SavedModelBundle> bundle,
    std::unique_ptr<RegressorInterface>* service) {
  service->reset(new SavedModelRegressor(run_options, std::move(bundle)));
  return OkStatus();
}

Status CreateFlyweightTensorFlowRegressor(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    std::unique_ptr<RegressorInterface>* service) {
  return CreateFlyweightTensorFlowRegressor(
      run_options, session, signature, thread::ThreadPoolOptions(), service);
}

Status CreateFlyweightTensorFlowRegressor(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    const thread::ThreadPoolOptions& thread_pool_options,
    std::unique_ptr<RegressorInterface>* service) {
  service->reset(new SavedModelTensorFlowRegressor(
      run_options, session, signature, thread_pool_options));
  return OkStatus();
}

Status GetRegressionSignatureDef(const ModelSpec& model_spec,
                                 const MetaGraphDef& meta_graph_def,
                                 SignatureDef* signature) {
  const string signature_name = model_spec.signature_name().empty()
                                    ? kDefaultServingSignatureDefKey
                                    : model_spec.signature_name();
  auto iter = meta_graph_def.signature_def().find(signature_name);
  if (iter == meta_graph_def.signature_def().end()) {
    return errors::InvalidArgument(strings::StrCat(
        "No signature was found with the name: ", signature_name));
  }
  if (GetSignatureMethodNameCheckFeature()) {
    if (iter->second.method_name() != kRegressMethodName) {
      return errors::InvalidArgument(strings::StrCat(
          "Expected regression signature method_name to be ",
          kRegressMethodName, ". Was: ", iter->second.method_name()));
    }
  } else {
    TF_RETURN_IF_ERROR(PreProcessRegression(iter->second, nullptr, nullptr));
  }
  *signature = iter->second;
  return OkStatus();
}

Status PreProcessRegression(const SignatureDef& signature,
                            string* input_tensor_name,
                            std::vector<string>* output_tensor_names) {
  if (GetSignatureMethodNameCheckFeature() &&
      signature.method_name() != kRegressMethodName) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected regression signature method_name to be ", kRegressMethodName,
        ". Was: ", signature.method_name()));
  }
  if (signature.inputs().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one input Tensor."));
  }
  if (signature.outputs().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one output Tensor."));
  }

  auto input_iter = signature.inputs().find(kRegressInputs);
  if (input_iter == signature.inputs().end()) {
    return errors::InvalidArgument(
        "No regression inputs found in SignatureDef: ",
        signature.DebugString());
  }
  if (input_tensor_name != nullptr) {
    *input_tensor_name = input_iter->second.name();
  }

  auto output_iter = signature.outputs().find(kRegressOutputs);
  if (output_iter == signature.outputs().end()) {
    return errors::InvalidArgument(
        "No regression outputs found in SignatureDef: ",
        signature.DebugString());
  }
  if (output_tensor_names != nullptr) {
    output_tensor_names->push_back(output_iter->second.name());
  }
  return OkStatus();
}

Status PostProcessRegressionResult(
    const SignatureDef& signature, int num_examples,
    const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, RegressionResult* result) {
  if (output_tensors.size() != output_tensor_names.size()) {
    return errors::InvalidArgument(
        "Expected output_tensors and output_tensor_names to have the same "
        "size.");
  }

  auto output_iter = signature.outputs().find(kRegressOutputs);
  if (output_iter == signature.outputs().end()) {
    return errors::FailedPrecondition(
        "No regression outputs found in SignatureDef: ",
        signature.DebugString());
  }
  const string output_tensor_name = output_iter->second.name();
  const Tensor* output_tensor = nullptr;
  for (int i = 0; i < output_tensor_names.size(); ++i) {
    if (output_tensor_names[i] == output_tensor_name) {
      output_tensor = &output_tensors[i];
      break;
    }
  }

  // Ensure the regression score output is shaped how we expect.
  if (output_tensor == nullptr) {
    return errors::InvalidArgument(strings::StrCat(
        "Could not find output tensor '", output_tensor_name, "'"));
  }
  if (!(output_tensor->dims() == 1 ||
        (output_tensor->dims() == 2 && output_tensor->dim_size(1) == 1))) {
    return errors::InvalidArgument(
        "Expected output Tensor shape to be either [batch_size] or ",
        "[batch_size, 1] but got ", output_tensor->shape().DebugString());
  }
  if (num_examples != output_tensor->dim_size(0)) {
    return errors::InvalidArgument(strings::StrCat(
        "Input batch size did not match output batch size: ", num_examples,
        " vs. ", output_tensor->dim_size(0)));
  }
  if (output_tensor->dtype() != DT_FLOAT) {
    return errors::InvalidArgument("Expected output Tensor of DT_FLOAT.  Got: ",
                                   DataType_Name(output_tensor->dtype()));
  }

  if (output_tensor->NumElements() != num_examples) {
    return errors::InvalidArgument("Expected output batch size to be ",
                                   num_examples,
                                   ".  Got: ", output_tensor->NumElements());
  }

  const auto& output_tensor_flat = output_tensor->flat<float>();
  for (int i = 0; i < num_examples; ++i) {
    result->add_regressions()->set_value(output_tensor_flat(i));
  }
  return OkStatus();
}

Status RunRegress(const RunOptions& run_options,
                  const MetaGraphDef& meta_graph_def,
                  const absl::optional<int64_t>& servable_version,
                  Session* session, const RegressionRequest& request,
                  RegressionResponse* response,
                  const thread::ThreadPoolOptions& thread_pool_options) {
  SignatureDef signature;
  TF_RETURN_IF_ERROR(GetRegressionSignatureDef(request.model_spec(),
                                               meta_graph_def, &signature));

  std::unique_ptr<RegressorInterface> regressor_interface;
  TF_RETURN_IF_ERROR(CreateFlyweightTensorFlowRegressor(
      run_options, session, &signature, thread_pool_options,
      &regressor_interface));

  MakeModelSpec(request.model_spec().name(),
                request.model_spec().signature_name(), servable_version,
                response->mutable_model_spec());

  // Run regression
  return regressor_interface->Regress(request, response->mutable_result());
}

}  // namespace serving
}  // namespace tensorflow
