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

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {

Status TensorflowPredictImpl::Predict(ServerCore* core,
                                      const PredictRequest& request,
                                      PredictResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }

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
  TF_RETURN_IF_ERROR(
      bundle->session->Run(inputs, output_tensor_names, {}, &outputs));

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

}  // namespace serving
}  // namespace tensorflow
