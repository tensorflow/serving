/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/simple_servers.h"

#include <algorithm>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/dynamic_manager.h"
#include "tensorflow_serving/core/eager_unload_policy.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/core/version_policy.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {
namespace simple_servers {

namespace {

// Creates a DynamicManager with the EagerUnloadPolicy.
std::unique_ptr<DynamicManager> CreateDynamicManager() {
  DynamicManager::Options manager_options;
  manager_options.version_policy.reset(new EagerUnloadPolicy());
  std::unique_ptr<DynamicManager> manager(
      new DynamicManager(std::move(manager_options)));
  return manager;
}

// Creates a Source<StoragePath> that monitors a filesystem's base_path for new
// directories. Upon finding these, it provides the target with the new version
// (a directory). The servable_name param simply allows this source to create
// all AspiredVersions for the target with the same servable_name.
Status CreateStoragePathSource(
    const string& base_path, const string& servable_name,
    Target<StoragePath>* target,
    std::unique_ptr<Source<StoragePath>>* path_source) {
  FileSystemStoragePathSourceConfig config;
  config.set_servable_name(servable_name);
  config.set_base_path(base_path);
  config.set_file_system_poll_wait_seconds(1);

  std::unique_ptr<FileSystemStoragePathSource> file_system_source;
  TF_RETURN_IF_ERROR(
      FileSystemStoragePathSource::Create(config, &file_system_source));

  ConnectSourceToTarget(file_system_source.get(), target);

  *path_source = std::move(file_system_source);
  return Status::OK();
}

// Creates a SessionBundle Source by adapting the underlying
// FileSystemStoragePathSource. These two are connected in the
// 'CreateSingleTFModelManagerFromBasePath' method, with the
// FileSystemStoragePathSource as the Source and the SessionBundleSource as the
// Target.
Status CreateSessionBundleSource(
    DynamicManager* manager,
    std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
        source) {
  SessionBundleSourceAdapterConfig config;

  source->reset(new SessionBundleSourceAdapter(config));

  ConnectSourceToTarget(source->get(), manager);

  return Status::OK();
}

}  // namespace

Status CreateSingleTFModelManagerFromBasePath(
    const string& base_path, UniquePtrWithDeps<Manager>* manager) {
  std::unique_ptr<DynamicManager> dynamic_manager = CreateDynamicManager();

  std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>
      bundle_source;
  TF_RETURN_IF_ERROR(
      CreateSessionBundleSource(dynamic_manager.get(), &bundle_source));

  std::unique_ptr<Source<StoragePath>> path_source;
  TF_RETURN_IF_ERROR(CreateStoragePathSource(
      base_path, "default", bundle_source.get(), &path_source));

  manager->SetOwned(std::move(dynamic_manager));
  manager->AddDependency(std::move(bundle_source));
  manager->AddDependency(std::move(path_source));

  return Status::OK();
}

}  // namespace simple_servers
}  // namespace serving
}  // namespace tensorflow
