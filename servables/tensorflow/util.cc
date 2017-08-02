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

#include "tensorflow_serving/servables/tensorflow/util.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/input.pb.h"

namespace tensorflow {
namespace serving {

int NumInputExamples(const Input& input) {
  switch (input.kind_case()) {
    case Input::KindCase::kExampleList:
      return input.example_list().examples_size();
    case Input::KindCase::kExampleListWithContext:
      return input.example_list_with_context().examples_size();
    default:
      break;
  }
  return 0;
}

Status InputToSerializedExampleTensor(const Input& input, Tensor* examples) {
  const int n = NumInputExamples(input);
  *examples = Tensor(DT_STRING, TensorShape({n}));
  switch (input.kind_case()) {
    case Input::KindCase::KIND_NOT_SET:
      break;

    case Input::KindCase::kExampleList: {
      auto input_vec = examples->vec<string>();
      int input_vec_index = 0;
      for (const auto& entry : input.example_list().examples()) {
        input_vec(input_vec_index++) = entry.SerializeAsString();
      }
      break;
    }

    case Input::KindCase::kExampleListWithContext: {
      const string context =
          input.example_list_with_context().context().SerializeAsString();
      auto input_vec = examples->vec<string>();
      int input_vec_index = 0;
      for (const auto& entry : input.example_list_with_context().examples()) {
        // Avoid the need for repeated serialization of context by simply
        // appending the Example serialization to the pre-serialized context.
        input_vec(input_vec_index++) =
            strings::StrCat(context, entry.SerializeAsString());
      }
    } break;

    default:
      return errors::Unimplemented("Input with kind ", input.kind_case(),
                                   " not supported.");
  }
  return Status::OK();
}

Status PerformOneShotTensorComputation(
    const Input& input, const string& input_tensor_name,
    const std::vector<string>& output_tensor_names, Session* session,
    std::vector<Tensor>* outputs) {
  // Setup the input Tensor to be a vector of string containing the serialized
  // tensorflow.Example.
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(InputToSerializedExampleTensor(input, &input_tensor));

  return session->Run({{input_tensor_name, input_tensor}}, output_tensor_names,
                      {}, outputs);
}

}  // namespace serving
}  // namespace tensorflow
