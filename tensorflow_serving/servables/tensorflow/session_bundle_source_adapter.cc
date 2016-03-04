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

#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"

#include <algorithm>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace {

SessionOptions GetSessionOptions(const SessionBundleConfig& config) {
  SessionOptions options;
  options.target = config.session_target();
  options.config = config.session_config();
  return options;
}

}  // namespace

SessionBundleSourceAdapter::SessionBundleSourceAdapter(
    const SessionBundleSourceAdapterConfig& config)
    : SimpleLoaderSourceAdapter<StoragePath, SessionBundle>([config](
          const StoragePath& path, std::unique_ptr<SessionBundle> * bundle) {
        bundle->reset(new SessionBundle);
        TF_RETURN_IF_ERROR(LoadSessionBundleFromPath(
            GetSessionOptions(config.config()), path, bundle->get()));
        (*bundle)->session.reset(
            new ServingSessionWrapper(std::move((*bundle)->session)));
        return Status::OK();
      }) {}

}  // namespace serving
}  // namespace tensorflow
