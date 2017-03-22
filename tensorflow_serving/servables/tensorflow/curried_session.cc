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

#include "tensorflow_serving/servables/tensorflow/curried_session.h"

namespace tensorflow {
namespace serving {

CurriedSession::CurriedSession(
    std::unique_ptr<Session> wrapped,
    const std::vector<std::pair<string, Tensor>>& curried_inputs)
    : wrapped_(std::move(wrapped)), curried_inputs_(curried_inputs) {
  for (const auto& entry : curried_inputs) {
    const string& name = entry.first;
    curried_input_names_.insert(name);
  }
}

Status CurriedSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                           const std::vector<string>& output_tensor_names,
                           const std::vector<string>& target_node_names,
                           std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(ValidateExplicitInputsDontMatchCurriedInputs(inputs));
  const std::vector<std::pair<string, Tensor>> combined_inputs =
      AddCurriedInputs(inputs);
  return wrapped_->Run(combined_inputs, output_tensor_names, target_node_names,
                       outputs);
}

Status CurriedSession::Run(const RunOptions& run_options,
                           const std::vector<std::pair<string, Tensor>>& inputs,
                           const std::vector<string>& output_tensor_names,
                           const std::vector<string>& target_node_names,
                           std::vector<Tensor>* outputs,
                           RunMetadata* run_metadata) {
  TF_RETURN_IF_ERROR(ValidateExplicitInputsDontMatchCurriedInputs(inputs));
  const std::vector<std::pair<string, Tensor>> combined_inputs =
      AddCurriedInputs(inputs);
  return wrapped_->Run(run_options, combined_inputs, output_tensor_names,
                       target_node_names, outputs, run_metadata);
}

Status CurriedSession::ValidateExplicitInputsDontMatchCurriedInputs(
    const std::vector<std::pair<string, Tensor>>& explicit_inputs) const {
  for (const auto& entry : explicit_inputs) {
    const string& name = entry.first;
    if (curried_input_names_.find(name) != curried_input_names_.end()) {
      return errors::InvalidArgument(
          "Explicit Run() input has same name as curried input ", name);
    }
  }
  return Status::OK();
}

std::vector<std::pair<string, Tensor>> CurriedSession::AddCurriedInputs(
    const std::vector<std::pair<string, Tensor>>& explicit_inputs) const {
  std::vector<std::pair<string, Tensor>> combined_inputs = explicit_inputs;
  std::copy(curried_inputs_.begin(), curried_inputs_.end(),
            std::back_inserter(combined_inputs));
  return combined_inputs;
}

}  // namespace serving
}  // namespace tensorflow
