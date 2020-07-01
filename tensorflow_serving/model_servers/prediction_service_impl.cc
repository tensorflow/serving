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
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference_helper.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"

#include <chrono>

namespace tensorflow {
namespace serving {

bvar::LatencyRecorder PredictionServiceImpl::predict_latency_recorder("predict_statistics");
bvar::LatencyRecorder PredictionServiceImpl::classify_latency_recorder("classify_statistics");
bvar::LatencyRecorder PredictionServiceImpl::regress_latency_recorder("regress_statistics");
bvar::LatencyRecorder PredictionServiceImpl::multiinference_latency_recorder("multiinference_statistics");

namespace {

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

thread::ThreadPoolOptions GetThreadPoolOptions(
    ThreadPoolFactory *thread_pool_factory) {
  thread::ThreadPoolOptions thread_pool_options;
  if (thread_pool_factory != nullptr) {
    thread_pool_options.inter_op_threadpool =
        thread_pool_factory->GetInterOpThreadPool();
    thread_pool_options.intra_op_threadpool =
        thread_pool_factory->GetIntraOpThreadPool();
  }
  return thread_pool_options;
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

  auto start = std::chrono::system_clock::now();
  const ::grpc::Status status =
      ToGRPCStatus(predictor_->Predict(run_options, core_, *request, response));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  predict_latency_recorder<<elapsed_seconds.count()*1000000;

  if (!status.ok()) {
    VLOG(1) << "Predict failed: " << status.error_message();
  }
  return status;
}

::grpc::Status PredictionServiceImpl::GetModelMetadata(
    ::grpc::ServerContext *context, const GetModelMetadataRequest *request,
    GetModelMetadataResponse *response) {
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
  auto start = std::chrono::system_clock::now();
  const ::grpc::Status status =
      ToGRPCStatus(TensorflowClassificationServiceImpl::Classify(
          run_options, core_, GetThreadPoolOptions(thread_pool_factory_),
          *request, response));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  classify_latency_recorder<<elapsed_seconds.count()*1000000;
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
  auto start = std::chrono::system_clock::now();
  const ::grpc::Status status =
      ToGRPCStatus(TensorflowRegressionServiceImpl::Regress(
          run_options, core_, GetThreadPoolOptions(thread_pool_factory_),
          *request, response));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  regress_latency_recorder<<elapsed_seconds.count()*1000000;
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
  auto start = std::chrono::system_clock::now();
  const ::grpc::Status status = ToGRPCStatus(RunMultiInferenceWithServerCore(
      run_options, core_, GetThreadPoolOptions(thread_pool_factory_), *request,
      response));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  multiinference_latency_recorder<<elapsed_seconds.count()*1000000;
  if (!status.ok()) {
    VLOG(1) << "MultiInference request failed: " << status.error_message();
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
