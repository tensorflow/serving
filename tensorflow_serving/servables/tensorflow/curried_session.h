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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_CURRIED_SESSION_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_CURRIED_SESSION_H_

#include "tensorflow_serving/servables/tensorflow/serving_session.h"

namespace tensorflow {
namespace serving {

// A session that wraps another session, while injecting a fixed set of
// additional input tensors into each Run() call. Useful for injecting static
// configuration tensors into a session without requiring the caller to be aware
// of them.
//
// It is an error to call Run() with an input that has the same name as one of
// the curried inputs.
class CurriedSession : public ServingSession {
 public:
  CurriedSession(std::unique_ptr<Session> wrapped,
                 const std::vector<std::pair<string, Tensor>>& curried_inputs);
  ~CurriedSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

 private:
  // Verifies no overlap between the tensor names in 'explicit_inputs' and
  // 'curried_inputs_'.
  Status ValidateExplicitInputsDontMatchCurriedInputs(
      const std::vector<std::pair<string, Tensor>>& explicit_inputs) const;

  // Adds 'curried_inputs_' to 'explicit_inputs'.
  std::vector<std::pair<string, Tensor>> AddCurriedInputs(
      const std::vector<std::pair<string, Tensor>>& explicit_inputs) const;

  // The session to which Run() calls are forwarded (after appending the
  // curried inputs).
  const std::unique_ptr<Session> wrapped_;

  // The inputs that get appended to 'inputs' in each Run() call.
  const std::vector<std::pair<string, Tensor>> curried_inputs_;

  // The tensor names from 'curried_inputs_'.
  std::set<string> curried_input_names_;

  TF_DISALLOW_COPY_AND_ASSIGN(CurriedSession);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_CURRIED_SESSION_H_
