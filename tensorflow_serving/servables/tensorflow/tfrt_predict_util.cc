/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tfrt_predict_util.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tracing.h"  // NOLINT
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tsl/platform/error_logging.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

// Validate the request and construct input tensor handles.
absl::Status PreProcessPredictionWithoutOutputFilter(
    const tfrt::FunctionMetadata& function_metadata,
    const PredictRequest& request, std::vector<Tensor>* input_tensors) {
  input_tensors->reserve(function_metadata.GetInputNames().size());
  for (int i = 0; i < function_metadata.GetInputNames().size(); ++i) {
    const auto& input_name = function_metadata.GetInputNames()[i];
    const auto input = request.inputs().find(input_name);
    if (input == request.inputs().end()) {
      const auto& default_inputs = function_metadata.GetDefaultInputs();
      const auto& default_input = default_inputs.find(input_name);
      if (default_input == default_inputs.end()) {
        const std::set<std::string> request_inputs =
            GetMapKeys(request.inputs());
        const std::set<std::string> required_inputs(
            function_metadata.GetInputNames().begin(),
            function_metadata.GetInputNames().end());
        const std::set<std::string> sent_extra =
            SetDifference(request_inputs, required_inputs);
        const std::set<std::string> missing =
            SetDifference(SetDifference(required_inputs, request_inputs),
                          saved_model::GetMapKeys(default_inputs));
        return absl::InvalidArgumentError(absl::StrCat(
            "Request inputs do not match required inputs for model `",
            request.model_spec().name(), "`. Send extra: {",
            absl::StrJoin(sent_extra, ","), "}. Missing but required: {",
            absl::StrJoin(missing, ","), "}."));
      }
      Tensor tensor;
      if (!tensor.FromProto(default_input->second)) {
        return absl::InvalidArgumentError(
            absl::StrCat("tensor parsing error: ", input_name));
      }
      input_tensors->emplace_back(std::move(tensor));
      continue;
    }
    Tensor tensor;
    if (!tensor.FromProto(input->second)) {
      return absl::InvalidArgumentError(
          absl::StrCat("tensor parsing error: ", input_name));
    }
    const auto expected_dtype = function_metadata.GetInputSpecs()[i].dtype;
    // TODO(b/188570937): Remove this type check and update related tests.
    if (expected_dtype != DT_INVALID  // Skip if the dtype is unspecified.
        && tensor.dtype() != expected_dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected input ", input_name, " to be ",
                       DataTypeString(expected_dtype), " but get ",
                       DataTypeString(tensor.dtype()), "."));
    }
    input_tensors->emplace_back(std::move(tensor));
  }
  return absl::OkStatus();
}

// Selects the outputs whose names are in `request.output_filter()`, preserving
// the order of `output_names`. The output filter must have been validated
// against `output_names` beforehand (see `ValidateOutputFilter`).
absl::Status SelectFilteredOutputs(const PredictRequest& request,
                                   const std::vector<std::string>& output_names,
                                   std::vector<Tensor> outputs,
                                   std::vector<std::string>* filtered_names,
                                   std::vector<Tensor>* filtered_outputs) {
  if (output_names.size() != outputs.size()) {
    return absl::UnknownError("Predict internal error.");
  }
  const absl::flat_hash_set<absl::string_view> output_filter(
      request.output_filter().begin(), request.output_filter().end());
  filtered_names->reserve(output_filter.size());
  filtered_outputs->reserve(output_filter.size());
  for (int i = 0; i < outputs.size(); ++i) {
    if (output_filter.contains(output_names[i])) {
      filtered_names->push_back(output_names[i]);
      filtered_outputs->push_back(std::move(outputs[i]));
    }
  }
  return absl::OkStatus();
}

bool IsOutputFilterEmptyOrFullSet(
    const PredictRequest& request,
    const tfrt::FunctionMetadata& function_metadata) {
  if (request.output_filter().empty()) return true;
  if (request.output_filter().size() !=
      function_metadata.GetOutputNames().size())
    return false;
  std::vector<absl::string_view> output_filter_names(
      request.output_filter().begin(), request.output_filter().end());
  std::vector<absl::string_view> func_output_names(
      function_metadata.GetOutputNames().begin(),
      function_metadata.GetOutputNames().end());
  std::sort(output_filter_names.begin(), output_filter_names.end());
  std::sort(func_output_names.begin(), func_output_names.end());
  return output_filter_names == func_output_names;
}

// Validates that every name in `request.output_filter()` is an output of the
// function. This is meant to be called before executing the function so that
// invalid requests fail fast without running the model.
//
// Both the output filter and the function outputs are expected to be small, so
// a linear scan is used to avoid any allocation on the success path.
absl::Status ValidateOutputFilter(
    const PredictRequest& request,
    const tfrt::FunctionMetadata& function_metadata) {
  const auto& func_output_names = function_metadata.GetOutputNames();
  const auto is_unknown = [&func_output_names](const std::string& name) {
    return std::find(func_output_names.begin(), func_output_names.end(),
                     name) == func_output_names.end();
  };
  if (std::none_of(request.output_filter().begin(),
                   request.output_filter().end(), is_unknown)) {
    return absl::OkStatus();
  }

  // Error path only: collect details for a helpful error message.
  std::vector<absl::string_view> unknown_names;
  for (const std::string& name : request.output_filter()) {
    if (is_unknown(name)) {
      unknown_names.push_back(name);
    }
  }
  std::vector<absl::string_view> sorted_output_names(func_output_names.begin(),
                                                     func_output_names.end());
  std::sort(sorted_output_names.begin(), sorted_output_names.end());
  return absl::InvalidArgumentError(
      absl::StrCat("output_filter contains non-existent output names: {",
                   absl::StrJoin(unknown_names, ","),
                   "}. Outputs expected to be in the set {",
                   absl::StrJoin(sorted_output_names, ","), "}."));
}

}  // namespace

namespace internal {
absl::Status RunPredict(
    const tfrt::SavedModel::RunOptions& run_options,
    const absl::optional<int64_t>& servable_version,
    const internal::PredictResponseTensorSerializationOption option,
    tfrt::SavedModel* saved_model, const PredictRequest& request,
    PredictResponse* response,
    const thread::ThreadPoolOptions& thread_pool_options) {
  // Validate signatures.
  const std::string function_name =
      request.model_spec().signature_name().empty()
          ? kDefaultServingSignatureDefKey
          : request.model_spec().signature_name();

  const auto function_metadata =
      saved_model->GetFunctionMetadata(function_name);
  if (!function_metadata.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Function \"", function_name, "\" not found."));
  }

  MakeModelSpec(request.model_spec().name(), function_name, servable_version,
                response->mutable_model_spec());

  auto run_opts = run_options;
  std::optional<tensorflow::tfrt_stub::TfThreadPoolWorkQueue> thread_pool;
  if (thread_pool_options.inter_op_threadpool != nullptr) {
    thread_pool.emplace(
        /*intra_op_threadpool=*/thread_pool_options.intra_op_threadpool,
        /*inter_op_threadpool=*/thread_pool_options.inter_op_threadpool);
    run_opts.work_queue = &(*thread_pool);
  }

  if (IsOutputFilterEmptyOrFullSet(request, function_metadata.value()) ||
      saved_model->disable_output_filter()) {
    TRACELITERAL("Pre process prediction without output filter");
    // When `disable_output_filter` is set, the whole function is executed and
    // the outputs are filtered afterwards. Validate the output filter before
    // execution so that invalid requests don't waste (e.g. TPU) compute.
    // Without `disable_output_filter`, only empty or full-set filters reach
    // here, which are always valid, so the check is skipped.
    if (saved_model->disable_output_filter() &&
        !request.output_filter().empty()) {
      TF_RETURN_IF_ERROR(
          ValidateOutputFilter(request, function_metadata.value()));
    }
    // Pre-processing.
    std::vector<Tensor> input_tensors;
    TF_RETURN_IF_ERROR(PreProcessPredictionWithoutOutputFilter(
        function_metadata.value(), request, &input_tensors));

    // Executes requests.
    TRACELITERAL("Execute prediction without output filter");
    std::vector<Tensor> outputs;
    const uint64_t start_microseconds = EnvTime::NowMicros();
    if (const auto status =
            saved_model->Run(run_opts, function_name, input_tensors, &outputs);
        !status.ok()) {
      if (IsTfrtErrorLoggingEnabled()) {
        tsl::error_logging::Log("TFRT", "SavedModelRun", status.message())
            .IgnoreError();
      }
      return status;
    }
    const uint64_t end_microseconds = EnvTime::NowMicros();
    RecordRuntimeLatency(request.model_spec().name(), /*api=*/"Predict",
                         /*runtime=*/"TFRT",
                         end_microseconds - start_microseconds);

    // Post-processing.
    TRACELITERAL("Post process prediction without output filter");
    const std::vector<std::string>& output_names =
        function_metadata->GetOutputNames();
    if (request.output_filter().empty()) {
      return PostProcessPredictionResult(output_names, outputs, option,
                                         response);
    }
    // The full signature was run, so keep only the outputs requested in
    // `output_filter`. It is known to be valid here: either it is the full set
    // or it was checked by `ValidateOutputFilter` before execution.
    std::vector<std::string> filtered_names;
    std::vector<Tensor> filtered_outputs;
    TF_RETURN_IF_ERROR(
        SelectFilteredOutputs(request, output_names, std::move(outputs),
                              &filtered_names, &filtered_outputs));
    return PostProcessPredictionResult(filtered_names, filtered_outputs, option,
                                       response);
  } else {
    // When output_filter is specified, use RunByTensorNames API to trigger
    // lazy initialization for optimized graph.
    // RunByTensorNames is discouraged for long run, we should consider to
    // deprecate output_filter and depends on different signature defs instead.
    const auto& metagraph_def = saved_model->GetMetaGraphDef();
    auto iter = metagraph_def.signature_def().find(function_name);
    if (iter == metagraph_def.signature_def().end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Serving signature key \"", function_name, "\" not found."));
    }
    const SignatureDef& signature = iter->second;

    TRACELITERAL("Pre process prediction with output filter");
    std::vector<std::pair<std::string, Tensor>> input_tensors;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> output_tensor_aliases;
    TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                            &output_tensor_names,
                                            &output_tensor_aliases));

    TRACELITERAL("Execute prediction with output filter");
    const uint64_t start_microseconds = EnvTime::NowMicros();
    std::vector<Tensor> outputs;
    if (const auto status = saved_model->RunByTensorNames(
            run_opts, input_tensors, output_tensor_names,
            /*target_node_names=*/{}, &outputs);
        !status.ok()) {
      if (IsTfrtErrorLoggingEnabled()) {
        tsl::error_logging::Log("TFRT", "SavedModelRun", status.message())
            .IgnoreError();
      }
      return status;
    }
    const uint64_t end_microseconds = EnvTime::NowMicros();
    RecordRuntimeLatency(request.model_spec().name(), /*api=*/"Predict",
                         /*runtime=*/"TFRT",
                         end_microseconds - start_microseconds);

    TRACELITERAL("Post process prediction with output filter");
    return PostProcessPredictionResult(output_tensor_aliases, outputs, option,
                                       response);
  }
}
}  // namespace internal

absl::Status RunPredict(const tfrt::SavedModel::RunOptions& run_options,
                        const absl::optional<int64_t>& servable_version,
                        tfrt::SavedModel* saved_model,
                        const PredictRequest& request,
                        PredictResponse* response,
                        const thread::ThreadPoolOptions& thread_pool_options) {
  return internal::RunPredict(
      run_options, servable_version,
      internal::PredictResponseTensorSerializationOption::kAsProtoField,
      saved_model, request, response, thread_pool_options);
}

}  // namespace serving
}  // namespace tensorflow
