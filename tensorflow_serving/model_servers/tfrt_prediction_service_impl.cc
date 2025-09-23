/* Copyright 2022 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/model_servers/tfrt_prediction_service_impl.h"

#include "grpc/grpc.h"
#include "grpcpp/server_context.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/model_servers/prediction_service_util.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

absl::Time TfrtPredictionServiceImpl::GetRequestDeadline(
    ::grpc::ServerContext *context) const {
  if (enforce_session_run_timeout_) {
    return absl::Now() +
           absl::Milliseconds(DeadlineToTimeoutMillis(context->raw_deadline()));
  }
  return absl::InfiniteFuture();
}

::grpc::Status TfrtPredictionServiceImpl::Predict(
    ::grpc::ServerContext *context, const PredictRequest *request,
    PredictResponse *response) {
  const uint64_t start = Env::Default()->NowMicros();

  Servable::RunOptions run_options;
  run_options.deadline = GetRequestDeadline(context);
  ServableHandle<Servable> servable;
  auto tf_status = core_->GetServableHandle(request->model_spec(), &servable);
  if (!tf_status.ok()) {
    VLOG(1) << "TFRT Predict get servable handle failed: "
            << tf_status.message();
    return ToGRPCStatus(tf_status);
  }

  tf_status = servable->Predict(run_options, *request, response);

  const ::grpc::Status status = ToGRPCStatus(tf_status);

  if (status.ok()) {
    RecordRequestLatency(request->model_spec().name(), /*api=*/"Predict",
                         /*entrypoint=*/"GRPC",
                         Env::Default()->NowMicros() - start);
  } else {
    VLOG(1) << "TFRT Predict failed: " << status.error_message();
  }
  RecordModelRequestCount(request->model_spec().name(), tf_status);

  return status;
}

::grpc::Status TfrtPredictionServiceImpl::GetModelMetadata(
    ::grpc::ServerContext *context, const GetModelMetadataRequest *request,
    GetModelMetadataResponse *response) {
  const absl::Status tf_status =
      TFRTGetModelMetadataImpl::GetModelMetadata(core_, *request, response);
  const ::grpc::Status status = ToGRPCStatus(tf_status);
  if (!status.ok()) {
    VLOG(1) << "TFRT GetModelMetadata failed: " << status.error_message();
  }
  return status;
}

::grpc::Status TfrtPredictionServiceImpl::Classify(
    ::grpc::ServerContext *context, const ClassificationRequest *request,
    ClassificationResponse *response) {
  const uint64_t start = Env::Default()->NowMicros();

  Servable::RunOptions run_options;
  run_options.deadline = GetRequestDeadline(context);
  ServableHandle<Servable> servable;
  auto tf_status = core_->GetServableHandle(request->model_spec(), &servable);
  if (!tf_status.ok()) {
    VLOG(1) << "TFRT Classify get servable handle failed: "
            << tf_status.message();
    return ToGRPCStatus(tf_status);
  }
  tf_status = servable->Classify(run_options, *request, response);

  const ::grpc::Status status = ToGRPCStatus(tf_status);

  if (status.ok()) {
    RecordRequestLatency(request->model_spec().name(), /*api=*/"Classify",
                         /*entrypoint=*/"GRPC",
                         Env::Default()->NowMicros() - start);
  } else {
    VLOG(1) << "TFRT Classify request failed: " << status.error_message();
  }
  RecordModelRequestCount(request->model_spec().name(), tf_status);

  return status;
}

::grpc::Status TfrtPredictionServiceImpl::Regress(
    ::grpc::ServerContext *context, const RegressionRequest *request,
    RegressionResponse *response) {
  const uint64_t start = Env::Default()->NowMicros();

  Servable::RunOptions run_options;
  run_options.deadline = GetRequestDeadline(context);
  ServableHandle<Servable> servable;
  auto tf_status = core_->GetServableHandle(request->model_spec(), &servable);
  if (!tf_status.ok()) {
    VLOG(1) << "TFRT Regress get servable handle failed: "
            << tf_status.message();
    return ToGRPCStatus(tf_status);
  }

  tf_status = servable->Regress(run_options, *request, response);

  const ::grpc::Status status = ToGRPCStatus(tf_status);

  if (status.ok()) {
    RecordRequestLatency(request->model_spec().name(), /*api=*/"Regress",
                         /*entrypoint=*/"GRPC",
                         Env::Default()->NowMicros() - start);
  } else {
    VLOG(1) << "TFRT Regress request failed: " << status.error_message();
  }
  RecordModelRequestCount(request->model_spec().name(), tf_status);

  return status;
}

namespace {

const ModelSpec &GetModelSpecFromRequest(const MultiInferenceRequest &request) {
  if (request.tasks_size() > 0 && request.tasks(0).has_model_spec()) {
    return request.tasks(0).model_spec();
  }
  return ModelSpec::default_instance();
}

}  // namespace

::grpc::Status TfrtPredictionServiceImpl::MultiInference(
    ::grpc::ServerContext *context, const MultiInferenceRequest *request,
    MultiInferenceResponse *response) {
  Servable::RunOptions run_options;
  run_options.deadline = GetRequestDeadline(context);
  ServableHandle<Servable> servable;

  auto tf_status =
      core_->GetServableHandle(GetModelSpecFromRequest(*request), &servable);
  if (!tf_status.ok()) {
    VLOG(1) << "TFRT MultiInference get model spec from request failed: "
            << tf_status.message();
    return ToGRPCStatus(tf_status);
  }

  tf_status = servable->MultiInference(run_options, *request, response);

  const ::grpc::Status status = ToGRPCStatus(tf_status);
  if (!status.ok()) {
    VLOG(1) << "TFRT MultiInference request failed: " << status.error_message();
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
