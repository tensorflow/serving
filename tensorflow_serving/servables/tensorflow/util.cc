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
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/threadpool_options.h"
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

// Metrics by model
auto* model_request_status_count_total = monitoring::Counter<2>::New(
    "/tensorflow/serving/model_request_count",
    "The total number of requests.", "model_name", "status");

auto* model_latency_total = monitoring::Counter<1>::New(
    "/tensorflow/serving/model_request_latency_usec",
    "The total time spent on executing graphs in microseconds.", "model_name");

auto* model_latency_histogram = monitoring::Sampler<1>::New(
    {"/tensorflow/serving/model_request_latency_histogram_usec",							    
     "The total time spent on executing graphs in microseconds.", "model_name"},
    // It would be nice to be able to set the parameters flexibly.
    monitoring::Buckets::Explicit({
	1000, 2000, 3000, 4000, 5000, 7000, 9000, 11000, 13000, 15000,
	17000, 19000, 21000, 24000, 27000, 30000, 33000, 35000, 38000}));

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

// Metrics by model
void RecordModelRequestCount(const string& model_name, const Status& status) {
  string status_label = "success";
  if (status != Status::OK()) {
    status_label = "failed";
  }
  model_request_status_count_total->GetCell(model_name, status_label)->IncrementBy(1);
}

void UpdateModelLatencyTime(const string& model_name, const uint64 running_time_usecs) {
  if (running_time_usecs > 0) {
    model_latency_total->GetCell(model_name)->IncrementBy(running_time_usecs);
    model_latency_histogram->GetCell(model_name)->Add(running_time_usecs);
  }
}


void RecordRequestExampleCount(const string& model_name, size_t count) {
  example_counts->GetCell(model_name)->Add(count);
  example_count_total->GetCell(model_name)->IncrementBy(count);
}

Status InputToSerializedExampleTensor(const Input& input, Tensor* examples) {
  internal::SerializedInput serialized_input;
  // There's a reason we serialize and then parse 'input' in this way:
  // 'example_list' and 'example_list_with_context' are lazily parsed
  // fields, which means they are lazily deserialized the very first
  // time they are accessed. So if we access them here for counting the
  // num_examples, then we'll pay a heavy cost of deserialization.
  //
  // SerializedInput proto has been created to prevent this, but at the same
  // time get the count of num_examples as well.
  bool parse_serialized_input_ok = false;
#if defined(PLATFORM_GOOGLE)
  // Benchmark ('BM_InputToSerializedExample') can help measure the effect of
  // changes in the future.
  parse_serialized_input_ok =
      serialized_input.ParseFromCord(input.SerializeAsCord());
#else
  parse_serialized_input_ok =
      serialized_input.ParseFromString(input.SerializeAsString());
#endif
  if (!parse_serialized_input_ok) {
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
      auto input_vec = examples->vec<tstring>();
      int input_vec_index = 0;
      for (const auto& entry : serialized_input.example_list().examples()) {
        input_vec(input_vec_index++) = entry;
      }
      break;
    }

    case Input::KindCase::kExampleListWithContext: {
      const auto& context =
          serialized_input.example_list_with_context().context();
      auto input_vec = examples->vec<tstring>();
      int input_vec_index = 0;
      for (const auto& entry :
           serialized_input.example_list_with_context().examples()) {
        tstring& input_str = input_vec(input_vec_index++);
        input_str.resize_uninitialized(context.size() + entry.size());
        // 'input_str_ptr' now points to the beginning of input_str.
        char* input_str_ptr = &input_str[0];
#if defined(PLATFORM_GOOGLE)
        // When absl::Cord OSS is fully shipped and protobuf open-source suports
        // Cord, we can get rid of marco above and unify code path.
        context.CopyToArray(input_str_ptr);
        entry.CopyToArray(input_str_ptr + context.size());
#else
        memcpy(input_str_ptr, &context[0], context.size());
        memcpy(input_str_ptr + context.size(), &entry[0], entry.size());
#endif
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
    std::vector<Tensor>* outputs, int* num_input_examples,
    const thread::ThreadPoolOptions& thread_pool_options) {
  // Setup the input Tensor to be a vector of string containing the serialized
  // tensorflow.Example.
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(InputToSerializedExampleTensor(input, &input_tensor));
  *num_input_examples = input_tensor.dim_size(0);

  RunMetadata run_metadata;
  return session->Run(run_options, {{input_tensor_name, input_tensor}},
                      output_tensor_names, {}, outputs, &run_metadata,
                      thread_pool_options);
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
