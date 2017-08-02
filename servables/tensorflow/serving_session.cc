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

#include "tensorflow_serving/servables/tensorflow/serving_session.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

Status ServingSession::Create(const GraphDef& graph) {
  return errors::PermissionDenied("State changes denied via ServingSession");
}

Status ServingSession::Extend(const GraphDef& graph) {
  return errors::PermissionDenied("State changes denied via ServingSession");
}

Status ServingSession::Close() {
  return errors::PermissionDenied("State changes denied via ServingSession");
}

}  // namespace serving
}  // namespace tensorflow
