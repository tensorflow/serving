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

#include <gmock/gmock.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"

namespace tensorflow {
namespace serving {

// A mock of tensorflow::serving::Servable.
class MockServable : public Servable {
 public:
  MockServable() : Servable("", 0) {}
  ~MockServable() override = default;

  MOCK_METHOD(absl::Status, Classify,
              (const tensorflow::serving::ClassificationRequest& request,
               tensorflow::serving::ClassificationResponse* response));
  MOCK_METHOD(absl::Status, Regress,
              (const tensorflow::serving::RegressionRequest& request,
               tensorflow::serving::RegressionResponse* response));
  MOCK_METHOD(absl::Status, Predict,
              (const tensorflow::serving::PredictRequest& request,
               tensorflow::serving::PredictResponse* response));
  MOCK_METHOD(absl::Status, PredictStreamed,
              (const tensorflow::serving::PredictRequest& request,
               absl::AnyInvocable<void(tensorflow::serving::PredictResponse)>
                   response_callback));
  MOCK_METHOD(absl::Status, MultiInference,
              (const tensorflow::serving::MultiInferenceRequest& request,
               tensorflow::serving::MultiInferenceResponse* response));
  MOCK_METHOD(absl::Status, GetModelMetadata,
              (const tensorflow::serving::GetModelMetadataRequest& request,
               tensorflow::serving::GetModelMetadataResponse* response));
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MOCK_SERVABLE_H_
