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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
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
        const std::set<string> request_inputs = GetMapKeys(request.inputs());
        const std::set<string> required_inputs(
            function_metadata.GetInputNames().begin(),
            function_metadata.GetInputNames().end());
        const std::set<string> sent_extra =
            SetDifference(request_inputs, required_inputs);
        const std::set<string> missing =
            SetDifference(SetDifference(required_inputs, request_inputs),
                          saved_model::GetMapKeys(default_inputs));
        return errors::InvalidArgument(absl::StrCat(
            "Request inputs do not match required inputs for model `",
            request.model_spec().name(), "`. Send extra: {",
            absl::StrJoin(sent_extra, ","), "}. Missing but required: {",
            absl::StrJoin(missing, ","), "}."));
      }
      Tensor tensor;
      if (!tensor.FromProto(default_input->second)) {
        return errors::InvalidArgument(
            absl::StrCat("tensor parsing error: ", input_name));
      }
      input_tensors->emplace_back(std::move(tensor));
      continue;
    }
    Tensor tensor;
    if (!tensor.FromProto(input->second)) {
      return errors::InvalidArgument(
          absl::StrCat("tensor parsing error: ", input_name));
    }
    const auto expected_dtype = function_metadata.GetInputSpecs()[i].dtype;
    // TODO(b/188570937): Remove this type check and update related tests.
    if (expected_dtype != DT_INVALID  // Skip if the dtype is unspecified.
        && tensor.dtype() != expected_dtype) {
      return errors::InvalidArgument(
          absl::StrCat("Expected input ", input_name, " to be ",
                       DataTypeString(expected_dtype), " but get ",
                       DataTypeString(tensor.dtype()), "."));
    }
    input_tensors->emplace_back(std::move(tensor));
  }
  return absl::OkStatus();
}

// Validate results and populate a PredictResponse.
// Tensors are serialized as specified.
absl::Status PostProcessPredictionResultWithoutOutputFilter(
    const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors,
    const internal::PredictResponseTensorSerializationOption option,
    const PredictRequest& request, PredictResponse* response) {
  if (output_tensor_names.size() != output_tensors.size()) {
    return errors::Unknown("Predict internal error.");
  }

  std::unordered_set<string> output_filter(request.output_filter().begin(),
                                           request.output_filter().end());
  int output_size = 0;
  for (int i = 0; i < output_tensors.size(); ++i) {
    if (!output_filter.empty() &&
        output_filter.find(output_tensor_names[i]) == output_filter.end()) {
      continue;
    }
    switch (option) {
      case internal::PredictResponseTensorSerializationOption::kAsProtoField: {
        output_tensors[i].AsProtoField(
            &((*response->mutable_outputs())[output_tensor_names[i]]));
      } break;
      case internal::PredictResponseTensorSerializationOption::
          kAsProtoContent: {
        output_tensors[i].AsProtoTensorContent(
            &((*response->mutable_outputs())[output_tensor_names[i]]));
      } break;
    }
    output_size++;
  }

  if (!output_filter.empty() && output_filter.size() != output_size) {
    return errors::InvalidArgument(absl::StrCat(
        "output_filter contains non-existed output names. output_filter: ",
        absl::StrJoin(output_filter, ",")));
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
    return errors::FailedPrecondition(
        strings::StrCat("Function \"", function_name, "\" not found."));
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

  if (IsOutputFilterEmptyOrFullSet(request, function_metadata.value())) {
    // Pre-processing.
    std::vector<Tensor> input_tensors;
    TF_RETURN_IF_ERROR(PreProcessPredictionWithoutOutputFilter(
        function_metadata.value(), request, &input_tensors));

    // Executes requests.
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
    return PostProcessPredictionResultWithoutOutputFilter(
        function_metadata->GetOutputNames(), outputs, option, request,
        response);
  } else {
    // When output_filter is specified, use RunByTensorNames API to trigger
    // lazy initialization for optimized graph.
    // RunByTensorNames is discouraged for long run, we should consider to
    // deprecate output_filter and depends on different signature defs instead.
    const auto& metagraph_def = saved_model->GetMetaGraphDef();
    auto iter = metagraph_def.signature_def().find(function_name);
    if (iter == metagraph_def.signature_def().end()) {
      return errors::FailedPrecondition(strings::StrCat(
          "Serving signature key \"", function_name, "\" not found."));
    }
    const SignatureDef& signature = iter->second;

    std::vector<std::pair<string, Tensor>> input_tensors;
    std::vector<string> output_tensor_names;
    std::vector<string> output_tensor_aliases;
    TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                            &output_tensor_names,
                                            &output_tensor_aliases));

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
