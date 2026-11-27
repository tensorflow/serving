/* Copyright 2018 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/predict_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

absl::Status VerifySignature(const SignatureDef& signature) {
  if (GetSignatureMethodNameCheckFeature() &&
      signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    return absl::InternalError(strings::StrCat(
        "Expected prediction signature method_name to be one of {",
        kPredictMethodName, ", ", kClassifyMethodName, ", ", kRegressMethodName,
        "}. Was: ", signature.method_name()));
  }
  return absl::OkStatus();
}

absl::Status VerifyRequestInputsSize(const SignatureDef& signature,
                                     const PredictRequest& request) {
  if (request.inputs().size() > signature.inputs().size() ||
      (request.inputs().size() < signature.inputs().size() &&
       signature.defaults().empty())) {
    const std::set<std::string> request_inputs = GetMapKeys(request.inputs());
    const std::set<std::string> signature_inputs =
        GetMapKeys(signature.inputs());
    const std::set<std::string> sent_extra =
        SetDifference(request_inputs, signature_inputs);
    const std::set<std::string> missing =
        SetDifference(signature_inputs, request_inputs);
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
        absl::StrCat(
            "input size does not match signature: ", request.inputs().size(),
            "!=", signature.inputs().size(), " len({",
            absl::StrJoin(request_inputs, ","), "}) != len({",
            absl::StrJoin(signature_inputs, ","), "}). Sent extra: {",
            absl::StrJoin(sent_extra, ","), "}. Missing but required: {",
            absl::StrJoin(missing, ","), "}."));
  }
  return absl::OkStatus();
}

}  // namespace

namespace internal {
absl::Status RunPredict(
    const RunOptions& run_options, const MetaGraphDef& meta_graph_def,
    const absl::optional<int64_t>& servable_version,
    const internal::PredictResponseTensorSerializationOption option,
    Session* session, const PredictRequest& request, PredictResponse* response,
    const thread::ThreadPoolOptions& thread_pool_options) {
  // Validate signatures.
  const absl::string_view signature_name =
      request.model_spec().signature_name().empty()
          ? kDefaultServingSignatureDefKey
          : absl::string_view(request.model_spec().signature_name());
  auto iter = meta_graph_def.signature_def().find(signature_name);
  if (iter == meta_graph_def.signature_def().end()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Serving signature key \"", signature_name, "\" not found."));
  }
  const SignatureDef& signature = iter->second;

  MakeModelSpec(request.model_spec().name(), signature_name, servable_version,
                response->mutable_model_spec());

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  std::vector<std::string> output_tensor_names;
  std::vector<std::string> output_tensor_aliases;
  TF_RETURN_IF_ERROR(PreProcessPrediction(signature, request, &input_tensors,
                                          &output_tensor_names,
                                          &output_tensor_aliases));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  const uint64_t start_microseconds = EnvTime::NowMicros();
  TF_RETURN_IF_ERROR(session->Run(run_options, input_tensors,
                                  output_tensor_names, {}, &outputs,
                                  &run_metadata, thread_pool_options));
  const uint64_t end_microseconds = EnvTime::NowMicros();
  RecordRuntimeLatency(request.model_spec().name(), /*api=*/"Predict",
                       /*runtime=*/"TF1",
                       end_microseconds - start_microseconds);

  return PostProcessPredictionResult(output_tensor_aliases, outputs, option,
                                     response);
}

absl::Status PreProcessPrediction(
    const SignatureDef& signature, const PredictRequest& request,
    std::vector<std::pair<std::string, Tensor>>* inputs,
    std::vector<std::string>* output_tensor_names,
    std::vector<std::string>* output_tensor_aliases) {
  TF_RETURN_IF_ERROR(VerifySignature(signature));
  TF_RETURN_IF_ERROR(VerifyRequestInputsSize(signature, request));
  inputs->reserve(inputs->size() + signature.inputs().size());
  TF_RETURN_IF_ERROR(
      saved_model::GetInputValues(signature, request.inputs(), *inputs));

  // Prepare run target.
  if (request.output_filter().empty()) {
    // When no output is specified, fetch all output tensors specified in
    // the signature.
    output_tensor_names->reserve(output_tensor_names->size() +
                                 signature.outputs().size());
    output_tensor_aliases->reserve(output_tensor_aliases->size() +
                                   signature.outputs().size());
    for (const auto& iter : signature.outputs()) {
      output_tensor_names->emplace_back(iter.second.name());
      output_tensor_aliases->emplace_back(iter.first);
    }
    return absl::OkStatus();
  }

  const int num_filters = request.output_filter().size();
  const size_t initial_aliases = output_tensor_aliases->size();
  output_tensor_names->reserve(output_tensor_names->size() + num_filters);
  output_tensor_aliases->reserve(initial_aliases + num_filters);
  // Typical PredictRequests filter 1 to 4 output aliases, and up to 8
  // multi-head outputs (e.g., scores, classes, logits, embeddings), a linear
  // scan over the contiguous `*output_tensor_aliases` vector is faster than
  // constructing and probing an `absl::flat_hash_set` on the heap; larger
  // filters use `seen_set` for O(1) lookup. Both closures fit in
  // `AnyInvocable`'s inline storage, so neither allocates.
  constexpr int kMaxLinearScanAliases = 8;
  absl::flat_hash_set<absl::string_view> seen_set;
  absl::AnyInvocable<bool(absl::string_view)> is_duplicate;
  if (num_filters <= kMaxLinearScanAliases) {
    is_duplicate = [output_tensor_aliases,
                    initial_aliases](absl::string_view alias) {
      return std::find(output_tensor_aliases->begin() + initial_aliases,
                       output_tensor_aliases->end(),
                       alias) != output_tensor_aliases->end();
    };
  } else {
    seen_set.reserve(num_filters);
    is_duplicate = [&seen_set](absl::string_view alias) {
      return !seen_set.insert(alias).second;
    };
  }
  for (const std::string& alias : request.output_filter()) {
    auto iter = signature.outputs().find(alias);
    if (iter == signature.outputs().end()) {
      return absl::Status(
          static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
          strings::StrCat("output tensor alias not found in signature: ", alias,
                          " Outputs expected to be in the set {",
                          absl::StrJoin(GetMapKeys(signature.outputs()), ","),
                          "}."));
    }
    if (is_duplicate(alias)) {
      return absl::Status(
          static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
          "duplicate output tensor alias: " + alias);
    }
    output_tensor_names->emplace_back(iter->second.name());
    output_tensor_aliases->emplace_back(alias);
  }
  return absl::OkStatus();
}

absl::Status PostProcessPredictionResult(
    const std::vector<std::string>& output_tensor_aliases,
    const std::vector<Tensor>& output_tensors,
    const internal::PredictResponseTensorSerializationOption option,
    PredictResponse* response) {
  // Validate and return output.
  if (output_tensors.size() != output_tensor_aliases.size()) {
    return absl::Status(
        static_cast<absl::StatusCode>(absl::StatusCode::kUnknown),
        "Predict internal error");
  }
  switch (option) {
    case internal::PredictResponseTensorSerializationOption::kAsProtoField: {
      for (int i = 0; i < output_tensors.size(); i++) {
        output_tensors[i].AsProtoField(
            &((*response->mutable_outputs())[output_tensor_aliases[i]]));
      }
    } break;
    case internal::PredictResponseTensorSerializationOption::kAsProtoContent: {
      for (int i = 0; i < output_tensors.size(); i++) {
        output_tensors[i].AsProtoTensorContent(
            &((*response->mutable_outputs())[output_tensor_aliases[i]]));
      }
    } break;
  }

  return absl::OkStatus();
}

}  // namespace internal

absl::Status RunPredict(const RunOptions& run_options,
                        const MetaGraphDef& meta_graph_def,
                        const absl::optional<int64_t>& servable_version,
                        Session* session, const PredictRequest& request,
                        PredictResponse* response,
                        const thread::ThreadPoolOptions& thread_pool_options) {
  return internal::RunPredict(
      run_options, meta_graph_def, servable_version,
      internal::PredictResponseTensorSerializationOption::kAsProtoField,
      session, request, response, thread_pool_options);
}

}  // namespace serving
}  // namespace tensorflow
