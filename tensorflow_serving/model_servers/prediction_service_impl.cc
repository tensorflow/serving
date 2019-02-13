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

#include "tensorflow_serving/model_servers/prediction_service_impl.h"

#include "grpc/grpc.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference_helper.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"

namespace tensorflow {
namespace serving {

namespace {

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

}  // namespace

::grpc::Status PredictionServiceImpl::Predict(::grpc::ServerContext *context,
                                              const PredictRequest *request,
                                              PredictResponse *response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  if (enforce_session_run_timeout_) {
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
  }

  const ::grpc::Status status =
      ToGRPCStatus(predictor_->Predict(run_options, core_, *request, response));

  if (!status.ok()) {
    VLOG(1) << "Predict failed: " << status.error_message();
  }
  return status;
}

::grpc::Status PredictionServiceImpl::GetModelMetadata(
    ::grpc::ServerContext *context, const GetModelMetadataRequest *request,
    GetModelMetadataResponse *response) {
  if (!use_saved_model_) {
    return ToGRPCStatus(
        errors::InvalidArgument("GetModelMetadata API is only available when "
                                "use_saved_model is set to true"));
  }
  const ::grpc::Status status = ToGRPCStatus(
      GetModelMetadataImpl::GetModelMetadata(core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "GetModelMetadata failed: " << status.error_message();
  }
  return status;
}

::grpc::Status PredictionServiceImpl::Classify(
    ::grpc::ServerContext *context, const ClassificationRequest *request,
    ClassificationResponse *response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  // By default, this is infinite which is the same default as RunOptions.
  if (enforce_session_run_timeout_) {
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
  }
  const ::grpc::Status status =
      ToGRPCStatus(TensorflowClassificationServiceImpl::Classify(
          run_options, core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "Classify request failed: " << status.error_message();
  }
  return status;
}

::grpc::Status PredictionServiceImpl::Regress(::grpc::ServerContext *context,
                                              const RegressionRequest *request,
                                              RegressionResponse *response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  // By default, this is infinite which is the same default as RunOptions.
  if (enforce_session_run_timeout_) {
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
  }
  const ::grpc::Status status =
      ToGRPCStatus(TensorflowRegressionServiceImpl::Regress(
          run_options, core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "Regress request failed: " << status.error_message();
  }
  return status;
}

::grpc::Status PredictionServiceImpl::MultiInference(
    ::grpc::ServerContext *context, const MultiInferenceRequest *request,
    MultiInferenceResponse *response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  // By default, this is infinite which is the same default as RunOptions.
  if (enforce_session_run_timeout_) {
    run_options.set_timeout_in_ms(
        DeadlineToTimeoutMillis(context->raw_deadline()));
  }
  const ::grpc::Status status = ToGRPCStatus(
      RunMultiInferenceWithServerCore(run_options, core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "MultiInference request failed: " << status.error_message();
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
