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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/predict_response_tensor_serialization_option.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// The RequestRecorder interface for implementations to inject custom metric and
// cost reporting.
class RequestRecorder {
 public:
  virtual ~RequestRecorder();
};

// Implements PredictionService`-like interface for a single SavedModel based on
// `tensorflow::tfrt_stub::SavedModel`. Executables are lazily compiled on its
// first use and cached. This class is thread-safe.
class TfrtSavedModelServable : public Servable {
 public:
  TfrtSavedModelServable(absl::string_view name, int64_t version,
                         const TfrtSavedModelConfig& config,
                         const SavedModelConfig& model_config,
                         std::unique_ptr<tfrt_stub::SavedModel> saved_model,
                         ThreadPoolFactory* thread_pool_factory)
      : TfrtSavedModelServable(
            name, version, config, model_config, std::move(saved_model),
            thread_pool_factory,
            [](TfrtSavedModelServable&) { return nullptr; }) {}

  TfrtSavedModelServable(
      absl::string_view name, int64_t version,
      const TfrtSavedModelConfig& config, const SavedModelConfig& model_config,
      std::unique_ptr<tfrt_stub::SavedModel> saved_model,
      ThreadPoolFactory* thread_pool_factory,
      std::function<std::unique_ptr<RequestRecorder>(TfrtSavedModelServable&)>
          recorder_creator);

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
      absl::AnyInvocable<void(absl::StatusOr<PredictResponse>)>
          response_callback) override;

  absl::Status MultiInference(const RunOptions& run_options,
                              const MultiInferenceRequest& request,
                              MultiInferenceResponse* response) override;

  absl::Status GetModelMetadata(const GetModelMetadataRequest& request,
                                GetModelMetadataResponse* response) override;

  bool SupportsPaging() const override { return true; }

  absl::Status Suspend() override;

  absl::Status Resume() override;

  tfrt_stub::SavedModel& saved_model() const { return *saved_model_; }

  void set_resume_fn(
      absl::AnyInvocable<absl::Status(TfrtSavedModelServable*)> resume_fn) {
    absl::MutexLock lock(&paging_mu_);
    resume_fn_ = std::move(resume_fn);
  }

  void set_suspend_fn(
      absl::AnyInvocable<absl::Status(TfrtSavedModelServable*)> suspend_fn) {
    absl::MutexLock lock(&paging_mu_);
    suspend_fn_ = std::move(suspend_fn);
  }

 private:
  tfrt_stub::SavedModel::RunOptions GetTFRTSavedModelRunOptions(
      const Servable::RunOptions& run_options) const;

  std::unique_ptr<RequestRecorder> CreateRecorder() {
    return recorder_creator_(*this);
  }

  std::unique_ptr<tfrt_stub::SavedModel> saved_model_;

  // `config_` is the adapter config, and it is the same for all
  // TfrtSavedModelServables within a model server.
  TfrtSavedModelConfig config_;

  internal::PredictResponseTensorSerializationOption
      predict_response_tensor_serialization_option_ =
          internal::PredictResponseTensorSerializationOption::kAsProtoField;

  // `thread_pool_factory_` is not owned by Servables. In a typical
  // implementation, the factory will own the `thread_pool_factory_` and it will
  // be shared across different Servables.
  ThreadPoolFactory* thread_pool_factory_ = nullptr;

  std::function<std::unique_ptr<RequestRecorder>(TfrtSavedModelServable&)>
      recorder_creator_ = [](TfrtSavedModelServable&) { return nullptr; };

  absl::AnyInvocable<absl::Status(TfrtSavedModelServable*)> suspend_fn_
      ABSL_GUARDED_BY(paging_mu_);

  absl::AnyInvocable<absl::Status(TfrtSavedModelServable*)> resume_fn_
      ABSL_GUARDED_BY(paging_mu_);

  bool suspended_ ABSL_GUARDED_BY(paging_mu_) = false;

  absl::Mutex paging_mu_;
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
