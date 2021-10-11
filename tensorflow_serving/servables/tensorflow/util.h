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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_

#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

// Implementation details mainly used for testing; please don't depend on it.
namespace internal {

monitoring::Sampler<1>* GetExampleCounts();

monitoring::Counter<1>* GetExampleCountTotal();

}  // namespace internal

// Metrics by model
void RecordModelRequestCount(const string& model_name, const Status& status);

// Enable/disable `method_name` checks on `SignatureDef` for predict, classify,
// regress APIs. Native TF2 models use fixed `method_name` for all APIs, and
// the check needs to be disabled to support both TF1 and (native) TF2 models.
//
// Disabling the check (typically done at process startup) should be OK and
// safe for most API users. By default the checks are enabled.
void SetSignatureMethodNameCheckFeature(bool v);

// Get current state of `method_name` check (see above for details).
bool GetSignatureMethodNameCheckFeature();

// Records the example count of this request with the metric tracking the
// histogram of number of examples per request.
void RecordRequestExampleCount(const string& model_name, size_t count);

// InputToSerializedExampleTensor populates a string Tensor of serialized
// Examples.
// If input has n Examples returns a string Tensor with shape {n}.
//
// Cases:
//   - Input kind unset: Tensor of shape {0}
//   - Input::example_list: Serializes each example.
//   - Input::example_list_with_context: Serializes each example merged with the
//     context.
//   - Other: non-OK Status.
//
// Note: does not perform any structural validation (e.g., if an example list is
// empty it will return a Tensor of shape {0}).
Status InputToSerializedExampleTensor(const Input& input, Tensor* examples);

// Issues a single Session::Run() call with 'input' to produce 'outputs'.
// Equivalent to InputToSerializedExampleTensor() followed by Session::Run().
Status PerformOneShotTensorComputation(
    const RunOptions& run_options, const Input& input,
    const string& input_tensor_name,
    const std::vector<string>& output_tensor_names, Session* session,
    std::vector<Tensor>* outputs, int* num_input_examples,
    const thread::ThreadPoolOptions& thread_pool_options =
        thread::ThreadPoolOptions(),
    int64_t* runtime_latency = nullptr);

// Same as PerformOneShotTensorComputation() above, except allows for multiple
// input tensor names (each tensor is fed the *same* `input`).
Status PerformOneShotTensorComputation(
    const RunOptions& run_options, const Input& input,
    const std::set<string>& input_tensor_names,
    const std::vector<string>& output_tensor_names, Session* session,
    std::vector<Tensor>* outputs, int* num_input_examples,
    const thread::ThreadPoolOptions& thread_pool_options =
        thread::ThreadPoolOptions());

// Populates given model_spec based on the model name and optional
// signature/version information.
// If signature_name has a value and is empty, model_spec's signature_name is
// set to tensorflow::kDefaultServingSignatureDefKey.
void MakeModelSpec(const string& model_name,
                   const absl::optional<string>& signature_name,
                   const absl::optional<int64_t>& version,
                   ModelSpec* model_spec);

// Gets the disk size of the model in the given path.
Status GetModelDiskSize(const string& path, FileProbingEnv* env,
                        uint64_t* total_file_size);

// Estimates the resources a session bundle or saved model bundle will use once
// loaded, from its export or saved model path. Directly uses disk state for
// estimation.
Status EstimateResourceFromPathUsingDiskState(const string& path,
                                              FileProbingEnv* env,
                                              ResourceAllocation* estimate);

// Update metrics for runtime latency.
void RecordRuntimeLatency(const string& model_name, const string& api,
                          const string& runtime, int64_t latency_usec);

// Update metrics for request latency.
void RecordRequestLatency(const string& model_name, const string& api,
                          const string& entrypoint, int64_t latency_usec);

// Get string keys of a map.
template <typename T>
std::set<string> GetMapKeys(const T& map) {
  std::set<string> keys;
  for (const auto& it : map) {
    keys.insert(it.first);
  }
  return keys;
}

// Returns a \ b, i.e. all items that are in `set_a` but not in `set_b`.
std::set<string> SetDifference(std::set<string> set_a, std::set<string> set_b);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_
