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

#include "tensorflow_serving/servables/tensorflow/predict_util.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

Status VerifySignature(const SignatureDef& signature) {
  if (signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    return errors::Internal(strings::StrCat(
        "Expected prediction signature method_name to be one of {",
        kPredictMethodName, ", ", kClassifyMethodName, ", ", kRegressMethodName,
        "}. Was: ", signature.method_name()));
  }
  return Status::OK();
}

template <typename T>
std::set<string> GetMapKeys(const T& proto_map) {
  std::set<string> keys;
  for (const auto& it : proto_map) {
    keys.insert(it.first);
  }
  return keys;
}

// Returns a \ b, i.e. all items that are in `set_a` but not in `set_b`.
std::set<string> SetDifference(std::set<string> set_a, std::set<string> set_b) {
  std::set<string> result;
  std::set_difference(set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
                      std::inserter(result, result.end()));
  return result;
}

Status VerifyRequestInputsSize(const SignatureDef& signature,
                               const PredictRequest& request) {
  if (request.inputs().size() != signature.inputs().size()) {
    const std::set<string> request_inputs = GetMapKeys(request.inputs());
    const std::set<string> signature_inputs = GetMapKeys(signature.inputs());
    const std::set<string> sent_extra =
        SetDifference(request_inputs, signature_inputs);
    const std::set<string> missing =
        SetDifference(signature_inputs, request_inputs);
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        absl::StrCat(
            "input size does not match signature: ", request.inputs().size(),
            "!=", signature.inputs().size(), " len({",
            absl::StrJoin(request_inputs, ","), "}) != len({",
            absl::StrJoin(signature_inputs, ","), "}). Sent extra: {",
            absl::StrJoin(sent_extra, ","), "}. Missing but required: {",
            absl::StrJoin(missing, ","), "}."));
  }
  return Status::OK();
}

// Validate a SignatureDef to make sure it's compatible with prediction, and
// if so, populate the input and output tensor names.
Status PreProcessPrediction(const SignatureDef& signature,
                            const PredictRequest& request,
                            std::vector<std::pair<string, Tensor>>* inputs,
                            std::vector<string>* output_tensor_names,
                            std::vector<string>* output_tensor_aliases) {
  TF_RETURN_IF_ERROR(VerifySignature(signature));
  TF_RETURN_IF_ERROR(VerifyRequestInputsSize(signature, request));
  for (auto& input : request.inputs()) {
    const string& alias = input.first;
    auto iter = signature.inputs().find(alias);
    if (iter == signature.inputs().end()) {
      return tensorflow::Status(
          tensorflow::error::INVALID_ARGUMENT,
          strings::StrCat("input tensor alias not found in signature: ", alias,
                          ". Inputs expected to be in the set {",
                          absl::StrJoin(GetMapKeys(signature.inputs()), ","),
                          "}."));
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
                          absl::StrJoin(GetMapKeys(signature.outputs()), ","),
                          "}."));
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
// Tensors are serialized as specified.
Status PostProcessPredictionResult(
    const std::vector<string>& output_tensor_aliases,
    const std::vector<Tensor>& output_tensors,
    const internal::PredictResponseTensorSerializationOption option,
    PredictResponse* response) {
  // Validate and return output.
  if (output_tensors.size() != output_tensor_aliases.size()) {
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "Predict internal error");
  }
  switch (option) {
    case internal::PredictResponseTensorSerializationOption::kAsProtoField: {
      for (int i = 0; i < output_tensors.size(); i++) {
        output_tensors[i].AsProtoField(
            &((*response->mutable_outputs())[output_tensor_aliases[i]]));
      }
    } break;
    case internal::PredictResponseTensorSerializationOption::kAsProtoContent: {
      for (int i = 0; i < output_tensors.size(); i++) {
        output_tensors[i].AsProtoTensorContent(
            &((*response->mutable_outputs())[output_tensor_aliases[i]]));
      }
    } break;
  }

  return Status::OK();
}

}  // namespace

namespace internal {
Status RunPredict(
    const RunOptions& run_options, const MetaGraphDef& meta_graph_def,
    const optional<int64>& servable_version,
    const internal::PredictResponseTensorSerializationOption option,
    Session* session, const PredictRequest& request,
    PredictResponse* response) {
  // Validate signatures.
  const string signature_name = request.model_spec().signature_name().empty()
                                    ? kDefaultServingSignatureDefKey
                                    : request.model_spec().signature_name();
  auto iter = meta_graph_def.signature_def().find(signature_name);
  if (iter == meta_graph_def.signature_def().end()) {
    return errors::FailedPrecondition(strings::StrCat(
        "Serving signature key \"", signature_name, "\" not found."));
  }
  const SignatureDef& signature = iter->second;

  MakeModelSpec(request.model_spec().name(), signature_name, servable_version,
                response->mutable_model_spec());

  std::vector<std::pair<string, Tensor>> input_tensors;
  std::vector<string> output_tensor_names;
  std::vector<string> output_tensor_aliases;
  TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                          &output_tensor_names,
                                          &output_tensor_aliases));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(session->Run(run_options, input_tensors,
                                  output_tensor_names, {}, &outputs,
                                  &run_metadata));

  return PostProcessPredictionResult(output_tensor_aliases, outputs, option,
                                     response);
}
}  // namespace internal

Status RunPredict(const RunOptions& run_options,
                  const MetaGraphDef& meta_graph_def,
                  const optional<int64>& servable_version, Session* session,
                  const PredictRequest& request, PredictResponse* response) {
  return internal::RunPredict(
      run_options, meta_graph_def, servable_version,
      internal::PredictResponseTensorSerializationOption::kAsProtoField,
      session, request, response);
}

}  // namespace serving
}  // namespace tensorflow
