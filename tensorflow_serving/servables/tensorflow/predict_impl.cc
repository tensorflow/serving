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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {
namespace {

// Implementation of Predict using the legacy SessionBundle GenericSignature.
Status SessionBundlePredict(const RunOptions& run_options, ServerCore* core,
                            const PredictRequest& request,
                            PredictResponse* response) {
  // Validate signatures.
  ServableHandle<SessionBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  Signature signature;
  TF_RETURN_IF_ERROR(
      GetNamedSignature("inputs", bundle->meta_graph_def, &signature));
  if (!signature.has_generic_signature()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "'inputs' named signature is not a generic signature");
  }
  GenericSignature input_signature = signature.generic_signature();
  TF_RETURN_IF_ERROR(
      GetNamedSignature("outputs", bundle->meta_graph_def, &signature));
  if (!signature.has_generic_signature()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "'outputs' named signature is not a generic signature");
  }
  GenericSignature output_signature = signature.generic_signature();

  // Verify and prepare input.
  if (request.inputs().size() != input_signature.map().size()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "input size does not match signature");
  }
  std::vector<std::pair<string, Tensor>> inputs;
  for (auto& input : request.inputs()) {
    const string& alias = input.first;
    auto iter = input_signature.map().find(alias);
    if (iter == input_signature.map().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          "input tensor alias not found in signature: " + alias);
    }
    Tensor tensor;
    if (!tensor.FromProto(input.second)) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "tensor parsing error: " + alias);
    }
    inputs.emplace_back(std::make_pair(iter->second.tensor_name(), tensor));
  }

  // Prepare run target.
  std::set<string> seen_outputs;
  std::vector<string> output_filter(request.output_filter().begin(),
                                    request.output_filter().end());
  std::vector<string> output_tensor_names;
  std::vector<string> output_aliases;
  for (auto& alias : output_filter) {
    auto iter = output_signature.map().find(alias);
    if (iter == output_signature.map().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          "output tensor alias not found in signature: " + alias);
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "duplicate output tensor alias: " + alias);
    }
    seen_outputs.insert(alias);
    output_tensor_names.emplace_back(iter->second.tensor_name());
    output_aliases.emplace_back(alias);
  }
  // When no output is specified, fetch all output tensors specified in
  // the signature.
  if (output_tensor_names.empty()) {
    for (auto& iter : output_signature.map()) {
      output_tensor_names.emplace_back(iter.second.tensor_name());
      output_aliases.emplace_back(iter.first);
    }
  }

  // Run session.
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(bundle->session->Run(
      run_options, inputs, output_tensor_names, {}, &outputs, &run_metadata));

  // Validate and return output.
  if (outputs.size() != output_tensor_names.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i].AsProtoField(
        &((*response->mutable_outputs())[output_aliases[i]]));
  }

  return Status::OK();
}

// Returns the keys in the map as a comma delimited string. Useful for debugging
// or when returning error messages.
// e.g. returns "key1, key2, key3".
string MapKeysToString(const google::protobuf::Map<string, tensorflow::TensorInfo>& map) {
  string result = "";
  for (const auto& i : map) {
    if (result.empty()) {
      result += i.first;
    } else {
      result += ", " + i.first;
    }
  }
  return result;
}

// Validate a SignatureDef to make sure it's compatible with prediction, and
// if so, populate the input and output tensor names.
Status PreProcessPrediction(const SignatureDef& signature,
                            const PredictRequest& request,
                            std::vector<std::pair<string, Tensor>>* inputs,
                            std::vector<string>* output_tensor_names,
                            std::vector<string>* output_tensor_aliases) {
  if (signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    return errors::Internal(strings::StrCat(
        "Expected prediction signature method_name to be one of {",
        kPredictMethodName, ", ", kClassifyMethodName, ", ", kRegressMethodName,
        "}. Was: ", signature.method_name()));
  }
  if (signature.inputs().empty()) {
    return errors::Internal(strings::StrCat(
        "Expected at least one input Tensor in prediction signature."));
  }
  if (signature.outputs().empty()) {
    return errors::Internal(strings::StrCat(
        "Expected at least one output Tensor in prediction signature."));
  }

  // Verify and prepare input.
  if (request.inputs().size() != signature.inputs().size()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "input size does not match signature");
  }
  for (auto& input : request.inputs()) {
    const string& alias = input.first;
    auto iter = signature.inputs().find(alias);
    if (iter == signature.inputs().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          strings::StrCat("input tensor alias not found in signature: ", alias,
                          ". Inputs expected to be in the set {",
                          MapKeysToString(signature.inputs()), "}."));
    }
    Tensor tensor;
    if (!tensor.FromProto(input.second)) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "tensor parsing error: " + alias);
    }
    inputs->emplace_back(std::make_pair(iter->second.name(), tensor));
  }

  // Prepare run target.
  std::set<string> seen_outputs;
  std::vector<string> output_filter(request.output_filter().begin(),
                                    request.output_filter().end());
  for (auto& alias : output_filter) {
    auto iter = signature.outputs().find(alias);
    if (iter == signature.outputs().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          strings::StrCat("output tensor alias not found in signature: ", alias,
                          " Outputs expected to be in the set {",
                          MapKeysToString(signature.outputs()), "}."));
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "duplicate output tensor alias: " + alias);
    }
    seen_outputs.insert(alias);
    output_tensor_names->emplace_back(iter->second.name());
    output_tensor_aliases->emplace_back(alias);
  }
  // When no output is specified, fetch all output tensors specified in
  // the signature.
  if (output_tensor_names->empty()) {
    for (auto& iter : signature.outputs()) {
      output_tensor_names->emplace_back(iter.second.name());
      output_tensor_aliases->emplace_back(iter.first);
    }
  }
  return Status::OK();
}

// Validate results and populate a PredictResponse.
Status PostProcessPredictionResult(
    const SignatureDef& signature,
    const std::vector<string>& output_tensor_aliases,
    const std::vector<Tensor>& output_tensors, PredictResponse* response) {
  // Validate and return output.
  if (output_tensors.size() != output_tensor_aliases.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }
  for (int i = 0; i < output_tensors.size(); i++) {
    output_tensors[i].AsProtoField(
        &((*response->mutable_outputs())[output_tensor_aliases[i]]));
  }
  return Status::OK();
}

// Implementation of Predict using the SavedModel SignatureDef format.
Status SavedModelPredict(const RunOptions& run_options, ServerCore* core,
                         const PredictRequest& request,
                         PredictResponse* response) {
  // Validate signatures.
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));

  const string signature_name = request.model_spec().signature_name().empty()
                                    ? kDefaultServingSignatureDefKey
                                    : request.model_spec().signature_name();
  auto iter = bundle->meta_graph_def.signature_def().find(signature_name);
  if (iter == bundle->meta_graph_def.signature_def().end()) {
    return errors::FailedPrecondition(
        "Default serving signature key not found.");
  }
  SignatureDef signature = iter->second;

  std::vector<std::pair<string, Tensor>> input_tensors;
  std::vector<string> output_tensor_names;
  std::vector<string> output_tensor_aliases;
  TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                          &output_tensor_names,
                                          &output_tensor_aliases));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(bundle->session->Run(run_options, input_tensors,
                                          output_tensor_names, {}, &outputs,
                                          &run_metadata));

  return PostProcessPredictionResult(signature, output_tensor_aliases, outputs,
                                     response);
}

}  // namespace

Status TensorflowPredictor::Predict(const RunOptions& run_options,
                                    ServerCore* core,
                                    const PredictRequest& request,
                                    PredictResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  if (use_saved_model_) {
    return SavedModelPredict(run_options, core, request, response);
  }
  return SessionBundlePredict(run_options, core, request, response);
}

}  // namespace serving
}  // namespace tensorflow
