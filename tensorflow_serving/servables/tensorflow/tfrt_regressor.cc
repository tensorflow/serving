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

#include "tensorflow_serving/servables/tensorflow/tfrt_regressor.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/error_logging.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/apis/regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

absl::Status PreProcessRegression(
    const tfrt::FunctionMetadata& function_metadata) {
  if (function_metadata.GetInputNames().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one input Tensor."));
  }
  if (function_metadata.GetOutputNames().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one output Tensor."));
  }

  if (function_metadata.GetInputNames()[0] != kRegressInputs) {
    return errors::FailedPrecondition(
        "No regression inputs found in function's metadata, only contains: ",
        function_metadata.GetInputNames()[0]);
  }

  if (function_metadata.GetOutputNames()[0] != kRegressOutputs) {
    return errors::FailedPrecondition(
        "No regression outputs found in function's metadata, only contains: ",
        function_metadata.GetOutputNames()[0]);
  }

  return absl::OkStatus();
}

absl::Status PostProcessRegressionResult(
    int num_examples, const std::vector<string>& output_tensor_names,
    const std::vector<Tensor>& output_tensors, RegressionResult* result) {
  if (output_tensors.size() != output_tensor_names.size()) {
    return errors::InvalidArgument(
        "Expected output_tensors and output_tensor_names to have the same "
        "size.");
  }

  const Tensor* output_tensor = &output_tensors[0];

  if (!(output_tensor->dims() == 1 ||
        (output_tensor->dims() == 2 && output_tensor->dim_size(1) == 1))) {
    return errors::InvalidArgument(
        "Expected output Tensor shape to be either [batch_size] or ",
        "[batch_size, 1] but got ", output_tensor->shape().DebugString());
  }
  if (num_examples != output_tensor->dim_size(0)) {
    return errors::InvalidArgument(strings::StrCat(
        "Input batch size did not match output batch size: ", num_examples,
        " vs. ", output_tensor->dim_size(0)));
  }
  if (output_tensor->dtype() != DT_FLOAT) {
    return errors::InvalidArgument("Expected output Tensor of DT_FLOAT.  Got: ",
                                   DataType_Name(output_tensor->dtype()));
  }

  if (output_tensor->NumElements() != num_examples) {
    return errors::InvalidArgument("Expected output batch size to be ",
                                   num_examples,
                                   ".  Got: ", output_tensor->NumElements());
  }

  const auto& output_tensor_flat = output_tensor->flat<float>();
  for (int i = 0; i < num_examples; ++i) {
    result->add_regressions()->set_value(output_tensor_flat(i));
  }
  return absl::OkStatus();
}

absl::Status RunRegress(const tfrt::SavedModel::RunOptions& run_options,
                        const absl::optional<int64_t>& servable_version,
                        tfrt::SavedModel* saved_model,
                        const RegressionRequest& request,
                        RegressionResponse* response) {
  const string function_name = request.model_spec().signature_name().empty()
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

  // Pre-processing.
  TF_RETURN_IF_ERROR(PreProcessRegression(function_metadata.value()));
  Tensor input_tensor;
  TF_RETURN_IF_ERROR(
      InputToSerializedExampleTensor(request.input(), &input_tensor));
  std::vector<Tensor> input_tensors;
  int num_examples = input_tensor.dim_size(0);
  input_tensors.emplace_back(std::move(input_tensor));

  // Executes requests.
  std::vector<Tensor> output_tensors;
  const uint64_t start_microseconds = EnvTime::NowMicros();
  if (const auto status = saved_model->Run(run_options, function_name,
                                           input_tensors, &output_tensors);
      !status.ok()) {
    if (IsTfrtErrorLoggingEnabled()) {
      tsl::error_logging::Log("TFRT", "SavedModelRun", status.message())
          .IgnoreError();
    }
    return status;
  }
  const uint64_t end_microseconds = EnvTime::NowMicros();
  RecordRuntimeLatency(request.model_spec().name(),
                       /*api=*/"Regress", /*runtime=*/"TFRT",
                       end_microseconds - start_microseconds);

  // Post-processing.
  return PostProcessRegressionResult(
      num_examples, function_metadata->GetOutputNames(), output_tensors,
      response->mutable_result());
}

}  // namespace serving
}  // namespace tensorflow
