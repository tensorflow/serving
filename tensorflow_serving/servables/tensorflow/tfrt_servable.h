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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SERVABLE_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SERVABLE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// Implements PredictionService`-like interface for a single SavedModel based on
// `tensorflow::tfrt_stub::SavedModel`. Executables are lazily compiled on its
// first use and cached. This class is thread-safe.
class TfrtSavedModelServable : public Servable {
 public:
  TfrtSavedModelServable(absl::string_view name, int64_t version,
                         const TfrtSavedModelConfig& config,
                         std::unique_ptr<tfrt_stub::SavedModel> saved_model,
                         ThreadPoolFactory* thread_pool_factory);

  absl::Status Classify(const RunOptions& run_options,
                        const ClassificationRequest& request,
                        ClassificationResponse* response) override;

  absl::Status Regress(const RunOptions& run_options,
                       const RegressionRequest& request,
                       RegressionResponse* response) override;

  absl::Status Predict(const RunOptions& run_options,
                       const PredictRequest& request,
                       PredictResponse* response) override;

  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> PredictStreamed(
      const RunOptions& run_options,
      absl::AnyInvocable<void(PredictResponse)> response_callback) override;

  absl::Status MultiInference(const RunOptions& run_options,
                              const MultiInferenceRequest& request,
                              MultiInferenceResponse* response) override;

  absl::Status GetModelMetadata(const GetModelMetadataRequest& request,
                                GetModelMetadataResponse* response) override;

  tfrt_stub::SavedModel& saved_model() const { return *saved_model_; }

 private:
  tfrt_stub::SavedModel::RunOptions GetTFRTSavedModelRunOptions(
      const Servable::RunOptions& run_options) const;

  std::unique_ptr<tfrt_stub::SavedModel> saved_model_;

  TfrtSavedModelConfig config_;

  internal::PredictResponseTensorSerializationOption
      predict_response_tensor_serialization_option_ =
          internal::PredictResponseTensorSerializationOption::kAsProtoField;

  // `thread_pool_factory_` is not owned by Servables. In a typical
  // implementation, the factory will own the `thread_pool_factory_` and it will
  // be shared across different Servables.
  ThreadPoolFactory* thread_pool_factory_ = nullptr;
};

// Creates a TfrtSavedModelServable from `saved_model_dir`.
absl::StatusOr<std::unique_ptr<TfrtSavedModelServable>>
CreateTfrtSavedModelServable(
    const tensorflow::tfrt_stub::SavedModel::Options& options,
    absl::string_view name, int64_t version, absl::string_view saved_model_dir,
    absl::flat_hash_set<std::string> tags);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SERVABLE_H_
