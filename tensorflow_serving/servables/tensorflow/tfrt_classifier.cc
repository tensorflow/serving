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

#include "tensorflow_serving/servables/tensorflow/tfrt_classifier.h"

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/error_logging.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/classifier.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

absl::Status PreProcessClassification(
    const tfrt::FunctionMetadata& function_metadata) {
  if (function_metadata.GetInputNames().size() != 1) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one input Tensor."));
  }
  if (function_metadata.GetOutputNames().size() != 1 &&
      function_metadata.GetOutputNames().size() != 2) {
    return errors::InvalidArgument(
        strings::StrCat("Expected one or two output Tensors, found ",
                        function_metadata.GetOutputNames().size()));
  }

  if (function_metadata.GetInputNames()[0] != kClassifyInputs) {
    return errors::FailedPrecondition(
        "No classification inputs found in function's metadata, only "
        "contains: ",
        function_metadata.GetInputNames()[0]);
  }

  bool find_output_classes = false;
  bool find_output_scores = false;
  for (const std::string& output_name : function_metadata.GetOutputNames()) {
    if (output_name == kClassifyOutputClasses) {
      find_output_classes = true;
    } else if ((output_name == kClassifyOutputScores)) {
      find_output_scores = true;
    }
  }

  if ((function_metadata.GetOutputNames().size() == 1 && !find_output_classes &&
       !find_output_scores) ||
      (function_metadata.GetOutputNames().size() == 2 &&
       !(find_output_classes && find_output_scores))) {
    return errors::FailedPrecondition(strings::StrCat(
        "Expected classification function outputs to contain", "\"",
        kClassifyOutputClasses, "\" and/or \"", kClassifyOutputScores, "\". "));
  }

  return absl::OkStatus();
}

absl::Status PostProcessClassificationResult(
    int num_examples, const std::vector<string>& output_names,
    const std::vector<Tensor>& output_tensors, ClassificationResult* result) {
  if (output_tensors.size() != output_names.size()) {
    return errors::InvalidArgument(strings::StrCat(
        "Unexpected output tensors size. Expected ", output_names.size(),
        " output tensor(s).  Got: ", output_tensors.size()));
  }

  const Tensor* classes = nullptr;
  const Tensor* scores = nullptr;
  for (int i = 0; i < output_tensors.size(); ++i) {
    if (output_names[i] == kClassifyOutputClasses) {
      classes = &output_tensors[i];
    } else if (output_names[i] == kClassifyOutputScores) {
      scores = &output_tensors[i];
    }
  }

  // Validate classes output Tensor.
  if (classes) {
    if (classes->dims() != 2) {
      return errors::InvalidArgument(
          "Expected Tensor shape: [batch_size num_classes] but got ",
          classes->shape().DebugString());
    }
    if (classes->dtype() != DT_STRING) {
      return errors::InvalidArgument(
          "Expected classes Tensor of DT_STRING. Got: ",
          DataType_Name(classes->dtype()));
    }
    if (classes->dim_size(0) != num_examples) {
      return errors::InvalidArgument("Expected classes output batch size of ",
                                     num_examples,
                                     ". Got: ", classes->dim_size(0));
    }
  }
  // Validate scores output Tensor.
  if (scores) {
    if (scores->dims() != 2) {
      return errors::InvalidArgument(
          "Expected Tensor shape: [batch_size num_classes] but got ",
          scores->shape().DebugString());
    }
    if (scores->dtype() != DT_FLOAT) {
      return errors::InvalidArgument(
          "Expected scores Tensor of DT_FLOAT. Got: ",
          DataType_Name(scores->dtype()));
    }
    if (scores->dim_size(0) != num_examples) {
      return errors::InvalidArgument("Expected scores output batch size of ",
                                     num_examples,
                                     ". Got: ", scores->dim_size(0));
    }
  }
  // Extract the number of classes from either the class or score output
  // Tensor.
  int num_classes = 0;
  if (classes && scores) {
    // If we have both Tensors they should agree in the second dimmension.
    if (classes->dim_size(1) != scores->dim_size(1)) {
      return errors::InvalidArgument(
          "Tensors class and score should match in dim_size(1). Got ",
          classes->dim_size(1), " vs. ", scores->dim_size(1));
    }
    num_classes = classes->dim_size(1);
  } else if (classes) {
    num_classes = classes->dim_size(1);
  } else if (scores) {
    num_classes = scores->dim_size(1);
  }

  // Convert the output to ClassificationResult format.
  for (int i = 0; i < num_examples; ++i) {
    serving::Classifications* classifications = result->add_classifications();
    for (int c = 0; c < num_classes; ++c) {
      serving::Class* cl = classifications->add_classes();
      if (classes) {
        const tstring& class_tstr = (classes->matrix<tstring>())(i, c);
        cl->set_label(class_tstr.data(), class_tstr.size());
      }
      if (scores) {
        cl->set_score((scores->matrix<float>())(i, c));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RunClassify(const tfrt::SavedModel::RunOptions& run_options,
                         const absl::optional<int64_t>& servable_version,
                         tfrt::SavedModel* saved_model,
                         const ClassificationRequest& request,
                         ClassificationResponse* response) {
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
  TF_RETURN_IF_ERROR(PreProcessClassification(function_metadata.value()));
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
  RecordRuntimeLatency(request.model_spec().name(), /*api=*/"Classify",
                       /*runtime=*/"TFRT",
                       end_microseconds - start_microseconds);

  // Post-processing.
  return PostProcessClassificationResult(
      num_examples, function_metadata->GetOutputNames(), output_tensors,
      response->mutable_result());
}

}  // namespace serving
}  // namespace tensorflow
