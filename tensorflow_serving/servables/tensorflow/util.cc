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

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/internal/serialized_input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {
namespace {

auto* example_counts = monitoring::Sampler<1>::New(
    {"/tensorflow/serving/request_example_counts",
     "The number of tensorflow.Examples per request.", "model"},
    // It's 15 buckets with the last bucket being 2^14 to DBL_MAX;
    // so the limits are [1, 2, 4, 8, ..., 16 * 1024, DBL_MAX].
    monitoring::Buckets::Exponential(1, 2, 15));

auto* example_count_total = monitoring::Counter<1>::New(
    "/tensorflow/serving/request_example_count_total",
    "The total number of tensorflow.Examples.", "model");

// Returns the number of examples in the Input.
int NumInputExamples(const internal::SerializedInput& input) {
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

}  // namespace

namespace internal {

monitoring::Sampler<1>* GetExampleCounts() { return example_counts; }

monitoring::Counter<1>* GetExampleCountTotal() { return example_count_total; }

}  // namespace internal

void RecordRequestExampleCount(const string& model_name, size_t count) {
  example_counts->GetCell(model_name)->Add(count);
  example_count_total->GetCell(model_name)->IncrementBy(count);
}

Status InputToSerializedExampleTensor(const Input& input, Tensor* examples) {
  const string serialized_input_str = input.SerializeAsString();
  internal::SerializedInput serialized_input;
  if (!serialized_input.ParseFromString(serialized_input_str)) {
    return errors::Internal("Error parsing serialized input.");
  }
  const int64 num_examples = NumInputExamples(serialized_input);
  if (num_examples == 0) {
    return errors::InvalidArgument("Input is empty.");
  }
  *examples = Tensor(DT_STRING, TensorShape({num_examples}));
  switch (serialized_input.kind_case()) {
    case Input::KindCase::KIND_NOT_SET:
      break;

    case Input::KindCase::kExampleList: {
      auto input_vec = examples->vec<string>();
      int input_vec_index = 0;
      for (const auto& entry : serialized_input.example_list().examples()) {
        input_vec(input_vec_index++) = entry;
      }
      break;
    }

    case Input::KindCase::kExampleListWithContext: {
      const string& context =
          serialized_input.example_list_with_context().context();
      auto input_vec = examples->vec<string>();
      int input_vec_index = 0;
      for (const auto& entry :
           serialized_input.example_list_with_context().examples()) {
        // Avoid the need for repeated serialization of context by simply
        // appending the Example serialization to the pre-serialized context.
        input_vec(input_vec_index++) = strings::StrCat(context, entry);
      }
    } break;

    default:
      return errors::Unimplemented(
          "Input with kind ", serialized_input.kind_case(), " not supported.");
  }
  return Status::OK();
}

Status PerformOneShotTensorComputation(
    const RunOptions& run_options, const Input& input,
    const string& input_tensor_name,
    const std::vector<string>& output_tensor_names, Session* session,
    std::vector<Tensor>* outputs, int* num_input_examples) {
  // Setup the input Tensor to be a vector of string containing the serialized
  // tensorflow.Example.
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(InputToSerializedExampleTensor(input, &input_tensor));
  *num_input_examples = input_tensor.dim_size(0);

  RunMetadata run_metadata;
  return session->Run(run_options, {{input_tensor_name, input_tensor}},
                      output_tensor_names, {}, outputs, &run_metadata);
}

void MakeModelSpec(const string& model_name,
                   const optional<string>& signature_name,
                   const optional<int64>& version, ModelSpec* model_spec) {
  model_spec->Clear();
  model_spec->set_name(model_name);
  if (signature_name) {
    model_spec->set_signature_name(signature_name->empty()
                                       ? kDefaultServingSignatureDefKey
                                       : *signature_name);
  }
  if (version) {
    model_spec->mutable_version()->set_value(*version);
  }
}

}  // namespace serving
}  // namespace tensorflow
