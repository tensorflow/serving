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
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
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

// Implementation of the RegressorInterface using the legacy SessionBundle
// RegressionSignature signature format.
class TensorFlowRegressor : public RegressorInterface {
 public:
  explicit TensorFlowRegressor(Session* session,
                               const RegressionSignature* const signature)
      : session_(session), signature_(signature) {}

  ~TensorFlowRegressor() override = default;

  Status Regress(const RegressionRequest& request,
                 RegressionResult* result) override {
    TRACELITERAL("TensorFlowRegressor::Regress");
    TRACELITERAL("ConvertInputTFEXamplesToTensor");
    // Setup the input Tensor to be a vector of string containing the serialized
    // tensorflow.Example.
    Tensor input_tensor;
    TF_RETURN_IF_ERROR(
        InputToSerializedExampleTensor(request.input(), &input_tensor));

    const int num_examples = input_tensor.dim_size(0);
    if (num_examples == 0) {
      return errors::InvalidArgument("RegressionRequest::input is empty.");
    }

    TRACELITERAL("RunRegression");
    Tensor output;
    TF_RETURN_IF_ERROR(
        RunRegression(*signature_, input_tensor, session_, &output));

    if (output.dtype() != DT_FLOAT) {
      return errors::Internal("Expected output Tensor of DT_FLOAT.  Got: ",
                              DataType_Name(output.dtype()));
    }

    if (output.NumElements() != num_examples) {
      return errors::Internal("Expected output batch size to be ", num_examples,
                              ".  Got: ", output.NumElements());
    }

    TRACELITERAL("ConvertToRegressionResult");
    for (int i = 0; i < num_examples; ++i) {
      result->add_regressions()->set_value(output.flat<float>()(i));
    }
    return Status::OK();
  }

 private:
  Session* const session_;
  const RegressionSignature* const signature_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorFlowRegressor);
};

// Implementation of the RegressorInterface using SavedModel.
class SavedModelTensorFlowRegressor : public RegressorInterface {
 public:
  explicit SavedModelTensorFlowRegressor(const RunOptions& run_options,
                                         Session* session,
                                         const SignatureDef* const signature)
      : run_options_(run_options), session_(session), signature_(signature) {}

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
    TF_RETURN_IF_ERROR(PerformOneShotTensorComputation(
        run_options_, request.input(), input_tensor_name, output_tensor_names,
        session_, &outputs, &num_examples));

    TRACELITERAL("ConvertToRegressionResult");
    return PostProcessRegressionResult(*signature_, num_examples,
                                       output_tensor_names, outputs, result);
  }

 private:
  const RunOptions run_options_;
  Session* const session_;
  const SignatureDef* const signature_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelTensorFlowRegressor);
};

// Implementation of the RegressorInterface
class SessionBundleRegressor : public RegressorInterface {
 public:
  explicit SessionBundleRegressor(std::unique_ptr<SessionBundle> bundle)
      : bundle_(std::move(bundle)) {}

  ~SessionBundleRegressor() override = default;

  Status Regress(const RegressionRequest& request,
                 RegressionResult* result) override {
    RegressionSignature signature;
    TF_RETURN_IF_ERROR(
        GetRegressionSignature(bundle_->meta_graph_def, &signature));

    TensorFlowRegressor regressor(bundle_->session.get(), &signature);
    return regressor.Regress(request, result);
  }

 private:
  std::unique_ptr<SessionBundle> bundle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SessionBundleRegressor);
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

Status CreateRegressorFromBundle(std::unique_ptr<SessionBundle> bundle,
                                 std::unique_ptr<RegressorInterface>* service) {
  service->reset(new SessionBundleRegressor(std::move(bundle)));
  return Status::OK();
}

Status CreateRegressorFromSavedModelBundle(
    const RunOptions& run_options, std::unique_ptr<SavedModelBundle> bundle,
    std::unique_ptr<RegressorInterface>* service) {
  service->reset(new SavedModelRegressor(run_options, std::move(bundle)));
  return Status::OK();
}

Status CreateFlyweightTensorFlowRegressor(
    Session* session, const RegressionSignature* const signature,
    std::unique_ptr<RegressorInterface>* service) {
  service->reset(new TensorFlowRegressor(session, signature));
  return Status::OK();
}

Status CreateFlyweightTensorFlowRegressor(
    const RunOptions& run_options, Session* session,
    const SignatureDef* signature,
    std::unique_ptr<RegressorInterface>* service) {
  service->reset(
      new SavedModelTensorFlowRegressor(run_options, session, signature));
  return Status::OK();
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
  if (iter->second.method_name() != kRegressMethodName) {
    return errors::InvalidArgument(strings::StrCat(
        "Expected regression signature method_name to be ", kRegressMethodName,
        ". Was: ", iter->second.method_name()));
  }
  *signature = iter->second;
  return Status::OK();
}

Status PreProcessRegression(const SignatureDef& signature,
                            string* input_tensor_name,
                            std::vector<string>* output_tensor_names) {
  if (signature.method_name() != kRegressMethodName) {
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
    return errors::FailedPrecondition(
        "No regression inputs found in SignatureDef: ",
        signature.DebugString());
  }
  *input_tensor_name = input_iter->second.name();

  auto output_iter = signature.outputs().find(kRegressOutputs);
  if (output_iter == signature.outputs().end()) {
    return errors::FailedPrecondition(
        "No regression outputs found in SignatureDef: ",
        signature.DebugString());
  }
  output_tensor_names->push_back(output_iter->second.name());
  return Status::OK();
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
  for (int i = 0; i < num_examples; ++i) {
    result->add_regressions()->set_value(output_tensor->flat<float>()(i));
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
