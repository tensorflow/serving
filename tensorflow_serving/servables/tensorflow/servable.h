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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVABLE_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVABLE_H_

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/run_options.h"

namespace tensorflow {
namespace serving {

inline constexpr absl::string_view kSignatureDef = "signature_def";

// Context of a `PredictStreamed` session. The caller of `PredictStreamed` calls
// `ProcessRequest` every time a request becomes available. The caller must call
// `Close()` at the end of the session before deleting the context object.
//
// The implementation can be thread-compatible. The caller is responsible for
// synchronizing all method invocations.
class PredictStreamedContext {
 public:
  virtual ~PredictStreamedContext() = default;

  // Consumes one incoming request. Blocking here may delay the consumption of
  // subsequent requests.
  virtual absl::Status ProcessRequest(const PredictRequest& request) = 0;

  // Closes the `PredictStreamed` session.
  virtual absl::Status Close() = 0;

  // Waits for all of the responses to be generated
  virtual absl::Status WaitResponses() = 0;
};

// A convenience wrapper for cases where the implementation allows exactly one
// request. `f` takes this single request and produces responses by calling the
// `response_callback` passed to `Servable::PredictStreamed`.
//
// This implementation is thread compatible but not thread safe.
class SingleRequestPredictStreamedContext final
    : public PredictStreamedContext {
 public:
  explicit SingleRequestPredictStreamedContext(
      absl::AnyInvocable<absl::Status(const PredictRequest&)> f);

  absl::Status ProcessRequest(const PredictRequest& request) final;
  absl::Status Close() final;
  absl::Status WaitResponses() final;

 private:
  absl::AnyInvocable<absl::Status(const PredictRequest&)> f_;
  bool one_request_received_ = false;
};

// Provides a `PredictionService`-like interface. All concrete implementations
// are expected to be thread-safe.
class Servable {
 public:
  Servable(absl::string_view name, int64_t version, bool is_critical = false)
      : name_(std::string(name)),
        version_(version),
        is_critical_(is_critical) {}

  virtual ~Servable() = default;

  // Returns the name associated with this servable.
  absl::string_view name() const { return name_; }

  // Returns the version associated with this servable.
  int64_t version() const { return version_; }

  bool IsCritical() const { return is_critical_; }

  using RunOptions = tensorflow::serving::servables::RunOptions;

  virtual absl::Status Classify(const RunOptions& run_options,
                                const ClassificationRequest& request,
                                ClassificationResponse* response) = 0;

  virtual absl::Status Regress(const RunOptions& run_options,
                               const RegressionRequest& request,
                               RegressionResponse* response) = 0;

  virtual absl::Status Predict(const RunOptions& run_options,
                               const PredictRequest& request,
                               PredictResponse* response) = 0;

  // Bidirectional streamed version of `Predict`. Returns a "context" object
  // that allows the caller to pass requests incrementally. The servable is kept
  // alive until the context object is deleted.
  //
  // `response_callback` is called for each streamed output, zero or more times,
  // when the streamed output becomes available. If an error is returned for any
  // response, subsequent responses and requests will be ignored and the error
  // will be returned. The callback invocation must be serialized by the
  // implementation, so that `response_callback` does not have to be
  // thread-safe, but blocking inside the callback may cause the next callback
  // invocation to be delayed. The implementation must guarantee that the
  // callback is never called after the `PredictStreamed` method returns.
  virtual absl::StatusOr<std::unique_ptr<PredictStreamedContext>>
  PredictStreamed(const RunOptions& run_options,
                  absl::AnyInvocable<void(absl::StatusOr<PredictResponse>)>
                      response_callback) = 0;

  virtual absl::Status MultiInference(const RunOptions& run_options,
                                      const MultiInferenceRequest& request,
                                      MultiInferenceResponse* response) = 0;

  virtual absl::Status GetModelMetadata(const GetModelMetadataRequest& request,
                                        GetModelMetadataResponse* response) = 0;

  // Returns true iff this servable supports paging.
  //
  // Paging is a process of moving model data (i.e., variables and executables)
  // between devices' HBM and host RAM. Servables that support paging can
  // time-share the available HBM and be paged in and out of the HBM according
  // to a paging policy.
  //
  // Note that even if a Servable supports paging, it is up to a Server
  // implementation to make active (or any!) use of the paging functionality.
  virtual bool SupportsPaging() const;

  // Pages out all variables and executables owned by this servable from
  // devices' HBM to host RAM.
  //
  // After this method returns, all requests return an error until `Resume()` is
  // called to bring the states back to device memory.
  //
  // If the suspension fails, the model is in an unspecified state and must be
  // unloaded and loaded again for it to be useful.
  //
  // This method may only be invoked if SupportsPaging() returns true.
  virtual absl::Status Suspend();

  // Inverse of `Suspend()`. Synchronously pages in all variables and
  // executables owned by this servable back to devices' HBM.
  //
  // Returns an error if the servable is not in a suspended state or resumption
  // failed. If the resumption fails, the model is in an unspecified state and
  // must be unloaded and loaded again for it to be useful.
  //
  // This method may only be invoked if SupportsPaging() returns true.
  virtual absl::Status Resume();

 private:
  // Metadata of this servable. Currently matches the fields in
  // `ServableId`.
  const std::string name_;
  const int64_t version_;
  const bool is_critical_;
};

// An "empty" servable where there's no model associated with the servable. All
// methods will return an error.
//
// Empty servables can be used in places where a servable is expected but we
// don't need to load any models. For example, Model Server currently expects
// each task to have at least one servable loaded, but Pathways Serving requires
// only the controller task to initiate loading servables. So we use empty
// servables in non-zero tasks to make sure non-zero tasks don't load anything.
class EmptyServable : public Servable {
 public:
  EmptyServable();

  absl::Status Classify(const RunOptions& run_options,
                        const ClassificationRequest& request,
                        ClassificationResponse* response) override {
    return error_;
  }

  absl::Status Regress(const RunOptions& run_options,
                       const RegressionRequest& request,
                       RegressionResponse* response) override {
    return error_;
  }

  absl::Status Predict(const RunOptions& run_options,
                       const PredictRequest& request,
                       PredictResponse* response) override {
    return error_;
  }

  absl::StatusOr<std::unique_ptr<PredictStreamedContext>> PredictStreamed(
      const RunOptions& run_options,
      absl::AnyInvocable<void(absl::StatusOr<PredictResponse>)>
          response_callback) {
    return error_;
  }

  absl::Status MultiInference(const RunOptions& run_options,
                              const MultiInferenceRequest& request,
                              MultiInferenceResponse* response) override {
    return error_;
  }

  absl::Status GetModelMetadata(const GetModelMetadataRequest& request,
                                GetModelMetadataResponse* response) override {
    return error_;
  }

 private:
  absl::Status error_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SERVABLE_H_
