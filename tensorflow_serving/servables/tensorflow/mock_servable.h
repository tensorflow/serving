/* Copyright 2023 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MOCK_SERVABLE_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MOCK_SERVABLE_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"

namespace tensorflow {
namespace serving {

class MockPredictStreamedContext : public PredictStreamedContext {
 public:
  MOCK_METHOD(absl::Status, ProcessRequest, (const PredictRequest& request),
              (final));
  MOCK_METHOD(absl::Status, Close, (), (final));
  MOCK_METHOD(absl::Status, WaitResponses, (), (final));
};

// A mock of tensorflow::serving::Servable.
class MockServable : public Servable {
 public:
  MockServable() : Servable("", 0) {}
  ~MockServable() override = default;

  MOCK_METHOD(absl::Status, Classify,
              (const tensorflow::serving::Servable::RunOptions& run_options,
               const tensorflow::serving::ClassificationRequest& request,
               tensorflow::serving::ClassificationResponse* response),
              (final));
  MOCK_METHOD(absl::Status, Regress,
              (const tensorflow::serving::Servable::RunOptions& run_options,
               const tensorflow::serving::RegressionRequest& request,
               tensorflow::serving::RegressionResponse* response),
              (final));
  MOCK_METHOD(absl::Status, Predict,
              (const tensorflow::serving::Servable::RunOptions& run_options,
               const tensorflow::serving::PredictRequest& request,
               tensorflow::serving::PredictResponse* response),
              (final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<PredictStreamedContext>>,
              PredictStreamed,
              (const tensorflow::serving::Servable::RunOptions& run_options,
               absl::AnyInvocable<
                   void(absl::StatusOr<tensorflow::serving::PredictResponse>)>
                   response_callback),
              (final));
  MOCK_METHOD(absl::Status, MultiInference,
              (const tensorflow::serving::Servable::RunOptions& run_options,
               const tensorflow::serving::MultiInferenceRequest& request,
               tensorflow::serving::MultiInferenceResponse* response),
              (final));
  MOCK_METHOD(absl::Status, GetModelMetadata,
              (const tensorflow::serving::GetModelMetadataRequest& request,
               tensorflow::serving::GetModelMetadataResponse* response),
              (final));
  MOCK_METHOD(bool, SupportsPaging, (), (const, final));
  MOCK_METHOD(absl::Status, Suspend, (), (final));
  MOCK_METHOD(absl::Status, Resume, (), (final));
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MOCK_SERVABLE_H_
