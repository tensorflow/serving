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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_SOURCE_ADAPTER_H_

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter that creates SessionBundle Loaders from export paths.
class SessionBundleSourceAdapter
    : public SimpleLoaderSourceAdapter<StoragePath, SessionBundle> {
 public:
  explicit SessionBundleSourceAdapter(
      const SessionBundleSourceAdapterConfig& config);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SessionBundleSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_SOURCE_ADAPTER_H_
