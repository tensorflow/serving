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

#include "tensorflow_serving/model_servers/platform_config_util.h"

#include "google/protobuf/any.pb.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

PlatformConfigMap CreateTensorFlowPlatformConfigMap(
    const SessionBundleConfig& session_bundle_config, bool use_saved_model) {
  PlatformConfigMap platform_config_map;
  ::google::protobuf::Any source_adapter_config;
  if (use_saved_model) {
    SavedModelBundleSourceAdapterConfig
        saved_model_bundle_source_adapter_config;
    *saved_model_bundle_source_adapter_config.mutable_legacy_config() =
        session_bundle_config;
    source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
  } else {
    SessionBundleSourceAdapterConfig session_bundle_source_adapter_config;
    *session_bundle_source_adapter_config.mutable_config() =
        session_bundle_config;
    source_adapter_config.PackFrom(session_bundle_source_adapter_config);
  }
  (*(*platform_config_map.mutable_platform_configs())[kTensorFlowModelPlatform]
        .mutable_source_adapter_config()) = source_adapter_config;
  return platform_config_map;
}

}  // namespace serving
}  // namespace tensorflow
