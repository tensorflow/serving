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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVING_SESSION_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVING_SESSION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace serving {

// A Session that blocks state-changing methods such as Close(), while allowing
// Run() for read-only access (not enforced). Useful for Session implementations
// that intend to be read-only and only implement Run().
class ServingSession : public Session {
 public:
  ServingSession() = default;
  ~ServingSession() override = default;

  // Methods that return errors.
  Status Create(const GraphDef& graph) final;
  Status Extend(const GraphDef& graph) final;
  Status Close() final;

  // (Subclasses just implement Run().)
};

// A ServingSession that wraps a given Session, and blocks all calls other than
// Run().
class ServingSessionWrapper : public ServingSession {
 public:
  explicit ServingSessionWrapper(std::unique_ptr<Session> wrapped)
      : wrapped_(std::move(wrapped)) {}
  ~ServingSessionWrapper() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override {
    return wrapped_->Run(inputs, output_tensor_names, target_node_names,
                         outputs);
  }

 private:
  std::unique_ptr<Session> wrapped_;

  TF_DISALLOW_COPY_AND_ASSIGN(ServingSessionWrapper);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVING_SESSION_H_
