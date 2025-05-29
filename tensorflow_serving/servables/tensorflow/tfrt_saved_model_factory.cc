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

#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_factory.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tsl/platform/casts.h"
#include "tensorflow_serving/batching/tfrt_saved_model_with_batching.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"
#include "tensorflow_serving/servables/tensorflow/machine_learning_metadata.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_warmup.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory_config.pb.h"
#include "tensorflow_serving/session_bundle/graph_rewriter.h"

namespace tensorflow {
namespace serving {
namespace {

using Batcher = SharedBatchScheduler<SavedModelBatchingTask>;

absl::Status WrapSavedModelForBatching(
    const BatchingParameters& batching_config,
    std::shared_ptr<Batcher> batch_scheduler,
    const std::vector<std::string>& function_names,
    std::unique_ptr<tfrt_stub::SavedModel>* saved_model) {
  LOG(INFO) << "Wrapping saved model to perform batch processing";

  if (batch_scheduler == nullptr) {
    return errors::Internal("batch_scheduler not set");
  }
  if (*saved_model == nullptr) {
    return errors::Internal("saved model not set");
  }

  auto queue_options =
      GetQueueOptions<tensorflow::serving::SavedModelBatchingTask>(
          batching_config,
          [](std::unique_ptr<tensorflow::serving::SavedModelBatchingTask>*
                 input_task,
             int open_batch_remaining_slot, int max_batch_size,
             std::vector<
                 std::unique_ptr<tensorflow::serving::SavedModelBatchingTask>>*
                 output_tasks) -> absl::Status {
            return SplitSavedModelInputTask(input_task,
                                            open_batch_remaining_slot,
                                            max_batch_size, output_tasks);
          });

  SavedModelBatchingOptions batching_saved_model_options;
  for (int allowed_batch_size : batching_config.allowed_batch_sizes()) {
    batching_saved_model_options.allowed_batch_sizes.push_back(
        allowed_batch_size);
  }

  batching_saved_model_options.pad_variable_length_inputs =
      batching_config.pad_variable_length_inputs();

  auto create_queue =
      [batch_scheduler, queue_options](
          std::function<void(std::unique_ptr<Batch<SavedModelBatchingTask>>)>
              process_batch_callback,
          std::unique_ptr<BatchScheduler<SavedModelBatchingTask>>* queue) {
        TF_RETURN_IF_ERROR(batch_scheduler->AddQueue(
            queue_options, process_batch_callback, queue));
        return absl::OkStatus();
      };
  std::vector<FuncNameWithBatchingSchedulerCreator>
      func_name_with_batching_scheduler_creator;
  func_name_with_batching_scheduler_creator.reserve(function_names.size());
  for (const std::string& function_name : function_names) {
    func_name_with_batching_scheduler_creator.push_back(
        {function_name, create_queue});
  }

  return CreateSavedModelWithBatching(batching_saved_model_options,
                                      func_name_with_batching_scheduler_creator,
                                      std::move(*saved_model), saved_model);
}

TfrtCompileOptions::TpuAllowUnpaddedBatch ToTpuAllowUnpaddedBatch(
    const TfrtSavedModelConfig::TpuUnpaddedBatchMode
        tpu_unpadded_batch_mode_enum) {
  switch (tpu_unpadded_batch_mode_enum) {
    case TfrtSavedModelConfig::UNPADDED_BATCH_AUTO:
      return TfrtCompileOptions::TpuAllowUnpaddedBatch::kAuto;
    case TfrtSavedModelConfig::UNPADDED_BATCH_ENFORCED:
      return TfrtCompileOptions::TpuAllowUnpaddedBatch::kEnforced;
    case TfrtSavedModelConfig::UNPADDED_BATCH_DISABLED:
    default:
      return TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled;
  }
}

absl::StatusOr<std::unique_ptr<TfrtSavedModelFactory>>
CreateDefaultTfrtSavedModelFactory(const TfrtSavedModelConfig& config) {
  TF_ASSIGN_OR_RETURN(auto batcher, CreateBatchSchedulerFromConfig(config));
  TF_ASSIGN_OR_RETURN(auto thread_pool_factory,
                      CreateThreadPoolFactoryFromConfig(config));

  return std::make_unique<TfrtSavedModelFactory>(
      config, batcher, std::move(thread_pool_factory));
}

}  // namespace

TfrtSavedModelFactory::~TfrtSavedModelFactory() = default;

absl::Status TfrtSavedModelFactory::Create(
    const TfrtSavedModelConfig& config,
    std::unique_ptr<TfrtSavedModelFactory>* factory) {
  auto create_fn = GetGlobalTfrtSavedModelFactoryRegistry().Get();
  if (!create_fn) {
    return absl::InternalError(
        "Missing create_fn for the TfrtSavedModelFactory.");
  }
  TF_ASSIGN_OR_RETURN(*factory, create_fn(config));
  return absl::OkStatus();
}

absl::Status TfrtSavedModelFactory::EstimateResourceRequirement(
    const std::string& path, ResourceAllocation* estimate) const {
  return EstimateResourceFromPath(
      path, config().resource_estimation_uses_validation_result(), estimate);
}

absl::StatusOr<tfrt::SavedModel::Options> CreateCommonSavedModelOptions(
    const TfrtSavedModelConfig& config, tfrt_stub::Runtime* runtime,
    const std::string& path,
    const std::unordered_set<std::string>& saved_model_tags,
    const tensorflow::MetaGraphDef& meta_graph_def,
    const std::string& model_name, int64_t model_version) {
  tfrt::SavedModel::Options options(runtime);

  // Register the right type of custom backend currently only requires setting
  // `use_ifrt`.
  options.graph_execution_options.use_ifrt = config.tfrt_use_ifrt();

  options.disable_output_filter = config.disable_output_filter();
  options.enable_lazy_loading =
      meta_graph_def.signature_def_size() > config.lazy_init_threshold();
  options.maybe_load_from_mla = config.maybe_load_from_mla();
  options.lazy_loading_use_graph_executor =
      config.lazy_loading_use_graph_executor();
  auto& compile_options = options.graph_execution_options.compile_options;
  compile_options.enable_grappler = config.enable_grappler();
  compile_options.graph_options = config.graph_options();
  if (config.enable_saved_model_config()) {
    TF_RETURN_IF_ERROR(LoadSavedModelConfig(
        path, options.graph_execution_options.compile_options.graph_options,
        options.graph_execution_options.runtime_config));
  }
  if (config.target_tpu()) {
    compile_options.device_target = TfrtDeviceInfraTarget::kTpurt;
  } else if (config.enable_tfrt_gpu()) {
    compile_options.device_target = TfrtDeviceInfraTarget::kGpu;
  } else {
    compile_options.device_target = TfrtDeviceInfraTarget::kCpu;
  }
  compile_options.hoist_invariant_ops = config.hoist_invariant_ops();
  compile_options.sink_in_invariant_ops = config.sink_in_invariant_ops();
  compile_options.cost_threshold = config.stream_merge_threshold();
  compile_options.merge_inter_dependent_streams =
      config.merge_inter_dependent_streams();
  compile_options.tpu_move_resource_gather_to_host =
      config.tpu_move_resource_gather_to_host();
  compile_options.tpu_gather_table_width_threshold_bytes =
      config.tpu_gather_table_width_threshold_bytes();
  compile_options.tpu_fuse_ops = config.use_fused_tpu_op();
  compile_options.enable_while_parallel_iterations =
      config.enable_while_parallel_iterations();
  compile_options.use_tpu_host_allocator_for_inputs =
      config.use_tpu_host_allocator_for_inputs();
  compile_options.tpu_allow_unpadded_batch =
      ToTpuAllowUnpaddedBatch(config.tpu_unpadded_batch_mode());
  compile_options.use_gpu_compile_and_execute_op =
      config.tfrt_use_fused_gpu_op();
  compile_options.min_num_batch_threads = config.tfrt_min_num_batch_threads();
  compile_options.min_max_enqueued_batches =
      config.tfrt_min_max_enqueued_batches();
  compile_options.batch_padding_policy = config.batch_padding_policy();
  compile_options.batch_options = config.in_graph_batching_parameters();

  options.graph_execution_options.run_placer_grappler_on_functions =
      config.run_placer_grappler_on_functions();
  options.graph_execution_options.enable_tfrt_gpu = config.enable_tfrt_gpu();
  options.graph_execution_options.tfrt_gpu_parallelism =
      config.tfrt_gpu_parallelism();
  options.graph_execution_options.gpu_system_memory_size_in_mb =
      config.gpu_system_memory_size_in_mb();
  options.graph_execution_options.enable_grappler_function_optimizer =
      config.enable_grappler_function_optimizer();
  options.graph_execution_options.enable_online_cost_analysis =
      config.enable_online_cost_analysis();
  options.graph_execution_options.enable_mlrt = config.enable_mlrt();
  options.graph_execution_options.model_metadata.set_name(model_name);
  options.graph_execution_options.model_metadata.set_version(model_version);
  return options;
}

absl::Status TfrtSavedModelFactory::CreateTfrtSavedModelWithMetadata(
    const Loader::Metadata& metadata, const std::string& path,
    std::unique_ptr<tfrt::SavedModel>* saved_model) {
  std::unordered_set<std::string> saved_model_tags(
      config().saved_model_tags().begin(), config().saved_model_tags().end());
  // Defaults to loading the meta graph def corresponding to the `serve` tag if
  // no `saved_model_tags` are specified.
  if (saved_model_tags.empty()) {
    saved_model_tags.insert(kSavedModelTagServe);
  }

  LOG(INFO) << "Creating TFRT SavedModel for path: " << path
            << " with config: " << config_.DebugString();
  auto* runtime = tensorflow::tfrt_stub::GetGlobalRuntime();
  tensorflow::MetaGraphDef meta_graph_def;
  TF_RETURN_IF_ERROR(tensorflow::ReadMetaGraphDefFromSavedModel(
      std::string(path), saved_model_tags, &meta_graph_def));
  if (auto& graph_rewriter = tensorflow::serving::GraphRewriter::GetGlobal();
      graph_rewriter.IsRegistered()) {
    TF_RETURN_IF_ERROR(graph_rewriter.Get()(&meta_graph_def));
  }
  TF_ASSIGN_OR_RETURN(
      tfrt::SavedModel::Options options,
      CreateCommonSavedModelOptions(config_, runtime, path, saved_model_tags,
                                    meta_graph_def, metadata.servable_id.name,
                                    metadata.servable_id.version));

  TF_RETURN_IF_ERROR(RegisterCustomBackend(options.graph_execution_options));

  TF_ASSIGN_OR_RETURN(*saved_model,
                      tfrt::SavedModelImpl::LoadSavedModel(
                          std::move(options), std::move(meta_graph_def), path));
  if (config_.has_batching_parameters() &&
      config_.batching_parameters().ByteSizeLong() != 0) {
    absl::optional<BatchingParameters> batching_params;
    TF_RETURN_IF_ERROR(GetPerModelBatchingParams(
        path, config_.batching_parameters(),
        config_.enable_per_model_batching_params(), &batching_params));
    if (batching_params.has_value()) {
      LOG(INFO) << "Wrapping TFRT SavedModel for batching with params: "
                << batching_params.value().DebugString();
      return WrapSavedModelForBatching(
          batching_params.value(), batch_scheduler_,
          (*saved_model)->GetFunctionNames(), saved_model);
    }
  }
  return absl::OkStatus();
}

absl::Status TfrtSavedModelFactory::CreateTfrtSavedModelWithMetadata(
    const Loader::Metadata& metadata, const std::string& path,
    std::unique_ptr<Servable>* servable) {
  TF_ASSIGN_OR_RETURN(auto override_servable, OverrideServable(metadata, path));
  if (override_servable) {
    *servable = std::move(override_servable);
    return absl::OkStatus();
  }

  std::unique_ptr<tfrt_stub::SavedModel> saved_model;
  TF_RETURN_IF_ERROR(
      CreateTfrtSavedModelWithMetadata(metadata, path, &saved_model));

  MaybePublishMLMDStreamz(path, metadata.servable_id.name,
                          metadata.servable_id.version);
  TF_ASSIGN_OR_RETURN(auto saved_model_config,
                      LoadSavedModelConfigOrDefault(path));

  *servable = std::make_unique<TfrtSavedModelServable>(
      metadata.servable_id.name, metadata.servable_id.version, config_,
      saved_model_config, std::move(saved_model), thread_pool_factory_.get(),
      recorder_creator_);
  TfrtSavedModelServable* tfrt_servable =
      down_cast<TfrtSavedModelServable*>(servable->get());

  if (config().enable_model_warmup()) {
    ModelWarmupOptions warmup_options = config().model_warmup_options();
    warmup_options.set_model_name(metadata.servable_id.name);
    warmup_options.set_model_version(metadata.servable_id.version);
    TF_RETURN_IF_ERROR(RunSavedModelWarmup(
        warmup_options, path, config().lazy_init_threshold(),
        config().skip_warmup_requests_if_initialized(),
        &tfrt_servable->saved_model()));
    if (config().freeze_after_init()) {
      TF_RETURN_IF_ERROR(Freeze(tfrt_servable->saved_model()));
    }
  }

  return absl::OkStatus();
}

TfrtSavedModelFactory::TfrtSavedModelFactory(
    const TfrtSavedModelConfig& config,
    std::shared_ptr<Batcher> batch_scheduler,
    std::unique_ptr<ThreadPoolFactory> thread_pool_factory,
    std::function<std::unique_ptr<RequestRecorder>(TfrtSavedModelServable&)>
        recorder_creator)
    : config_(config),
      batch_scheduler_(batch_scheduler),
      thread_pool_factory_(std::move(thread_pool_factory)),
      recorder_creator_(std::move(recorder_creator)) {}

TfrtSavedModelFactoryRegistry::TfrtSavedModelFactoryRegistry() {
  factory_create_fn_ = [](const TfrtSavedModelConfig& config) {
    return CreateDefaultTfrtSavedModelFactory(config);
  };
}

absl::string_view TfrtSavedModelFactory::GetServingResourceType() const {
  if (std::any_of(config_.saved_model_tags().begin(),
                  config_.saved_model_tags().end(),
                  [](const auto& tag) { return tag == kSavedModelTagTpu; })) {
    return device_types::kTpu;
  }
  if (std::any_of(config_.saved_model_tags().begin(),
                  config_.saved_model_tags().end(),
                  [](const auto& tag) { return tag == kSavedModelTagGpu; })) {
    return device_types::kGpu;
  }
  return device_types::kMain;
}

TfrtSavedModelFactoryRegistry& GetGlobalTfrtSavedModelFactoryRegistry() {
  static auto* const registry = new TfrtSavedModelFactoryRegistry;
  return *registry;
}

absl::StatusOr<std::shared_ptr<TfrtSavedModelFactory::Batcher>>
CreateBatchSchedulerFromConfig(const TfrtSavedModelConfig& config) {
  std::shared_ptr<Batcher> batcher;
  if (config.has_batching_parameters() &&
      config.batching_parameters().ByteSizeLong() != 0) {
    TF_RETURN_IF_ERROR(
        CreateBatchScheduler(config.batching_parameters(), &batcher));
  }
  return batcher;
}

absl::StatusOr<std::unique_ptr<ThreadPoolFactory>>
CreateThreadPoolFactoryFromConfig(const TfrtSavedModelConfig& config) {
  const auto& thread_pool_factory_config_filepath =
      config.thread_pool_factory_config_filepath();
  std::unique_ptr<ThreadPoolFactory> thread_pool_factory;
  if (!thread_pool_factory_config_filepath.empty()) {
    ThreadPoolFactoryConfig thread_pool_factory_config;
    TF_RETURN_IF_ERROR(tsl::ReadTextProto(tsl::Env::Default(),
                                          thread_pool_factory_config_filepath,
                                          &thread_pool_factory_config));
    TF_RETURN_IF_ERROR(ThreadPoolFactoryRegistry::CreateFromAny(
        thread_pool_factory_config.thread_pool_factory_config(),
        &thread_pool_factory));
  }
  return thread_pool_factory;
}

}  // namespace serving
}  // namespace tensorflow
