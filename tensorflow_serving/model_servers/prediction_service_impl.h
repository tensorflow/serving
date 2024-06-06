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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/prediction_service_util.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

namespace tensorflow {
namespace serving {

class PredictionServiceImpl final : public PredictionService::Service {
 public:
  explicit PredictionServiceImpl(const PredictionServiceOptions& options)
      : core_(options.server_core),
        predictor_(new TensorflowPredictor(options.thread_pool_factory)),
        enforce_session_run_timeout_(options.enforce_session_run_timeout),
        thread_pool_factory_(options.thread_pool_factory) {}

  ::grpc::Status Predict(::grpc::ServerContext* context,
                         const PredictRequest* request,
                         PredictResponse* response) override;

  ::grpc::Status GetModelMetadata(::grpc::ServerContext* context,
                                  const GetModelMetadataRequest* request,
                                  GetModelMetadataResponse* response) override;

  ::grpc::Status Classify(::grpc::ServerContext* context,
                          const ClassificationRequest* request,
                          ClassificationResponse* response) override;

  ::grpc::Status Regress(::grpc::ServerContext* context,
                         const RegressionRequest* request,
                         RegressionResponse* response) override;

  ::grpc::Status MultiInference(::grpc::ServerContext* context,
                                const MultiInferenceRequest* request,
                                MultiInferenceResponse* response) override;

 private:
  ServerCore* core_;
  std::unique_ptr<TensorflowPredictor> predictor_;
  const bool enforce_session_run_timeout_;
  ThreadPoolFactory* thread_pool_factory_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
