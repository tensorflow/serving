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

#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_source_adapter.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_util.h"
#include "tensorflow_serving/servables/tensorflow/file_acl.h"
#include "tensorflow_serving/servables/tensorflow/machine_learning_metadata.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_factory.h"

namespace tensorflow {
namespace serving {

// copybara:strip_begin (Do not leak in tensorflow serving OSS.)
namespace {
// Orbax manifest file name.
inline constexpr char kOrbaxModelManifestPb[] = "manifest.pb";
// Orbax manifest version file name.
inline constexpr char kOrbaxModelManifestVersionTxt[] = "manifest_version.txt";

absl::Status IsOrbaxModelDirectory(absl::string_view path) {
  const std::string orbax_model_manifest_pb_path =
      tensorflow::io::JoinPath(path, kOrbaxModelManifestPb);
  const std::string orbax_model_manifest_version_path =
      tensorflow::io::JoinPath(path, kOrbaxModelManifestVersionTxt);
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->FileExists(orbax_model_manifest_pb_path));
  TF_RETURN_IF_ERROR(env->FileExists(orbax_model_manifest_version_path));
  return absl::OkStatus();
}

}  // namespace
// copybara:strip_end

absl::Status TfrtSavedModelSourceAdapter::Create(
    const TfrtSavedModelSourceAdapterConfig& config,
    std::unique_ptr<TfrtSavedModelSourceAdapter>* adapter) {
  std::unique_ptr<TfrtSavedModelFactory> factory;
  TF_RETURN_IF_ERROR(
      TfrtSavedModelFactory::Create(config.saved_model_config(), &factory));
  adapter->reset(new TfrtSavedModelSourceAdapter(std::move(factory)));
  return absl::OkStatus();
}

TfrtSavedModelSourceAdapter::~TfrtSavedModelSourceAdapter() { Detach(); }

TfrtSavedModelSourceAdapter::TfrtSavedModelSourceAdapter(
    std::unique_ptr<TfrtSavedModelFactory> factory)
    : factory_(std::move(factory)) {}

SimpleLoader<Servable>::CreatorVariant
TfrtSavedModelSourceAdapter::GetServableCreator(
    std::shared_ptr<TfrtSavedModelFactory> factory,
    const StoragePath& path) const {
  return [factory, path](const Loader::Metadata& metadata,
                         std::unique_ptr<Servable>* servable) {
    TF_RETURN_IF_ERROR(RegisterModelRoot(metadata.servable_id, path));
    if (MaybeSavedModelDirectory(path)) {
      return factory->CreateTfrtSavedModelWithMetadata(metadata, path,
                                                       servable);
    }

    // copybara:strip_begin (Do not leak in tesorflow serving OSS.)
    if (IsOrbaxModelDirectory(path).ok()) {
      return factory->CreateOrbaxServable(metadata, path, servable);
    }
    // copybara:strip_end

    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported model directory: ", path,
                     ". Only SavedModel and Orbax Model are supported."));
  };
}

absl::Status TfrtSavedModelSourceAdapter::Convert(
    const StoragePath& path, std::unique_ptr<Loader>* loader) {
  std::shared_ptr<TfrtSavedModelFactory> factory = factory_;
  auto servable_creator = GetServableCreator(factory, path);
  auto resource_estimator = [factory, path](ResourceAllocation* estimate) {
    TF_RETURN_IF_ERROR(factory->EstimateResourceRequirement(path, estimate));

    ResourceUtil::Options resource_util_options;
    resource_util_options.devices = {{device_types::kMain, 1}};
    std::unique_ptr<ResourceUtil> resource_util =
        std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));
    const Resource ram_resource = resource_util->CreateBoundResource(
        device_types::kMain, resource_kinds::kRamBytes);
    resource_util->SetQuantity(
        ram_resource, resource_util->GetQuantity(ram_resource, *estimate),
        estimate);

    return absl::OkStatus();
  };
  auto post_load_resource_estimator = [factory,
                                       path](ResourceAllocation* estimate) {
    return factory->EstimateResourceRequirement(path, estimate);
  };
  loader->reset(new SimpleLoader<Servable>(servable_creator, resource_estimator,
                                           {post_load_resource_estimator}));
  return absl::OkStatus();
}

// Register the source adapter.
class TfrtSavedModelSourceAdapterCreator {
 public:
  static absl::Status Create(
      const TfrtSavedModelSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    std::unique_ptr<TfrtSavedModelFactory> factory;
    TF_RETURN_IF_ERROR(
        TfrtSavedModelFactory::Create(config.saved_model_config(), &factory));
    adapter->reset(new TfrtSavedModelSourceAdapter(std::move(factory)));
    return absl::OkStatus();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(TfrtSavedModelSourceAdapterCreator,
                                     TfrtSavedModelSourceAdapterConfig);

}  // namespace serving
}  // namespace tensorflow
