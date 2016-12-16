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

#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

// A SourceAdapter that creates SessionBundle Loaders from export paths. It
// keeps a SessionBundleFactory as its state, which may house a batch scheduler
// that is shared across all of the session bundles it emits.
class SessionBundleSourceAdapter final
    : public UnarySourceAdapter<StoragePath, std::unique_ptr<Loader>> {
 public:
  static Status Create(const SessionBundleSourceAdapterConfig& config,
                       std::unique_ptr<SessionBundleSourceAdapter>* adapter);

  ~SessionBundleSourceAdapter() override;

  // Returns a function to create a session bundle source adapter.
  static std::function<Status(
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*)>
  GetCreator(const SessionBundleSourceAdapterConfig& config);

 private:
  friend class SessionBundleSourceAdapterCreator;

  explicit SessionBundleSourceAdapter(
      std::unique_ptr<SessionBundleFactory> bundle_factory);

  Status Convert(const StoragePath& path,
                 std::unique_ptr<Loader>* loader) override;

  // We use a shared ptr to share ownership with Loaders we emit, in case they
  // outlive this object.
  std::shared_ptr<SessionBundleFactory> bundle_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(SessionBundleSourceAdapter);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_SOURCE_ADAPTER_H_
