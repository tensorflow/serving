/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_FACTORY_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/batching/tfrt_saved_model_with_batching.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// Create common saved model options for TFRT saved model.
absl::StatusOr<tfrt::SavedModel::Options> CreateCommonSavedModelOptions(
    const TfrtSavedModelConfig& config, tfrt_stub::Runtime* runtime,
    const std::string& path,
    const std::unordered_set<std::string>& saved_model_tags,
    const tensorflow::MetaGraphDef& meta_graph_def,
    const std::string& model_name, int64_t model_version);

/// A factory that creates tfrt_stub::SavedModel from SavedModel export paths.
///
/// The factory can also estimate the resource (e.g. RAM) requirements of a
/// tfrt_stub::SavedModel based on the SavedModel (i.e. prior to loading the
/// session).
///
/// This class is thread-safe.
class TfrtSavedModelFactory {
 public:
  using Batcher = SharedBatchScheduler<SavedModelBatchingTask>;

  TfrtSavedModelFactory(const TfrtSavedModelConfig& config,
                        std::shared_ptr<Batcher> batch_scheduler,
                        std::unique_ptr<ThreadPoolFactory> thread_pool_factory)
      : TfrtSavedModelFactory(config, std::move(batch_scheduler),
                              std::move(thread_pool_factory),
                              [](TfrtSavedModelServable&) { return nullptr; }) {
  }

  TfrtSavedModelFactory(
      const TfrtSavedModelConfig& config,
      std::shared_ptr<Batcher> batch_scheduler,
      std::unique_ptr<ThreadPoolFactory> thread_pool_factory,
      std::function<std::unique_ptr<RequestRecorder>(TfrtSavedModelServable&)>
          recorder_creator);

  virtual ~TfrtSavedModelFactory();

  /// Instantiates a TfrtSavedModelFactory using a config.
  ///
  /// @param config    Config with initialization options.
  /// @param factory   Newly created factory if the returned Status is OK.
  static absl::Status Create(const TfrtSavedModelConfig& config,
                             std::unique_ptr<TfrtSavedModelFactory>* factory);

  /// Instantiates a tfrt_stub::SavedModel from a given export or SavedModel
  /// path and the given metadata.
  ///
  /// @param metadata       Metadata to be associated with the saved_model.
  /// @param path           Path to the model.
  /// @param servable       Newly created Servable if the returned Status is OK.
  virtual absl::Status CreateTfrtSavedModelWithMetadata(
      const Loader::Metadata& metadata, const string& path,
      std::unique_ptr<Servable>* servable);

  ABSL_DEPRECATED("Use the overload that creates Servable instead")
  absl::Status CreateTfrtSavedModelWithMetadata(
      const Loader::Metadata& metadata, const string& path,
      std::unique_ptr<tfrt_stub::SavedModel>* saved_model);

  /// Estimates the resources a SavedModel will use once loaded, from its
  /// export path.
  ///
  /// @param path      Path to the model.
  /// @param estimate  Output resource usage estimates. Different kinds of
  /// resources (e.g. CPU, RAM, etc.) may get populated.
  absl::Status EstimateResourceRequirement(const string& path,
                                           ResourceAllocation* estimate) const;

  const TfrtSavedModelConfig& config() const { return config_; }
  TfrtSavedModelConfig& mutable_config() { return config_; }
  absl::string_view GetServingResourceType() const;

 private:
  // The subclass can override this method to return a custom servable
  // instead of creating one using CreateTfrtSavedModelWithMetadata(). If it
  // returns nullptr, CreateTfrtSavedModelWithMetadata() will be used normally.
  virtual absl::StatusOr<std::unique_ptr<Servable>> OverrideServable(
      const Loader::Metadata& metadata, const std::string& path) {
    return nullptr;
  }

  // The subclass can override this method to register the custom backend into
  // TFRT savedmodel.
  virtual absl::Status RegisterCustomBackend(
      tfrt_stub::GraphExecutionOptions& options) {
    return absl::OkStatus();
  }

  virtual absl::Status Freeze(tfrt_stub::SavedModel& saved_model) {
    return absl::OkStatus();
  }

  TfrtSavedModelConfig config_;

  // A shared batch scheduler. One queue is used for each saved model this
  // factory emits. If batching is not configured, this remains null.
  std::shared_ptr<Batcher> batch_scheduler_;

  // `thread_pool_factory_` is used to create inter-op ThreadPools. It can be a
  // nullptr and then the default Tensorflow threadpools should be used.
  std::unique_ptr<ThreadPoolFactory> thread_pool_factory_;

  std::function<std::unique_ptr<RequestRecorder>(TfrtSavedModelServable&)>
      recorder_creator_ = [](TfrtSavedModelServable&) { return nullptr; };

  TF_DISALLOW_COPY_AND_ASSIGN(TfrtSavedModelFactory);
};

// The registry for creating the TfrtSavedModelFactory. By default the CreateFn
// creates an instance of TfrtSavedModelFactory. Custom implementations can use
// this registry to override the CreateFn so that it creates an instance of the
// subclass of TfrtSavedModelFactory.
class TfrtSavedModelFactoryRegistry {
 public:
  using CreateFn =
      std::function<absl::StatusOr<std::unique_ptr<TfrtSavedModelFactory>>(
          const TfrtSavedModelConfig& config)>;

  TfrtSavedModelFactoryRegistry();

  void Register(CreateFn fn) {
    absl::MutexLock lock(&mu_);
    if (factory_create_fn_) {
      LOG(INFO) << "Overriding TfrtSavedModelFactory's create function.";
    }
    factory_create_fn_ = std::move(fn);
  }

  CreateFn Get() const {
    absl::MutexLock lock(&mu_);
    return factory_create_fn_;
  }

 private:
  mutable absl::Mutex mu_;
  CreateFn factory_create_fn_ ABSL_GUARDED_BY(mu_);
};

// Creates a batch scheduler based on `config`. The result can be a nullptr if
// `config` does not specify batch parameters.
absl::StatusOr<std::shared_ptr<TfrtSavedModelFactory::Batcher>>
CreateBatchSchedulerFromConfig(const TfrtSavedModelConfig& config);

// Creates a thread pool factory based on `config`. The result can be a nullptr
// if `config` does not specify thread pool factory config.
absl::StatusOr<std::unique_ptr<ThreadPoolFactory>>
CreateThreadPoolFactoryFromConfig(const TfrtSavedModelConfig& config);

TfrtSavedModelFactoryRegistry& GetGlobalTfrtSavedModelFactoryRegistry();

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_SAVED_MODEL_FACTORY_H_
