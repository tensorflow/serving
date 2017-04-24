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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_FACTORY_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/contrib/batching/shared_batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

// A factory that creates SavedModelBundles from SavedModel or SessionBundle
// export paths.
//
// The emitted sessions only support Run(), and although not enforced it is
// expected that the client will only make non-mutating Run() calls. (If this
// restriction, which we've added as a safety measure, is problematic for your
// use-case please contact the TensorFlow Serving team to discuss disabling it.)
//
// If the config calls for batching, the emitted sessions automatically batch
// Run() calls behind the scenes, using a SharedBatchScheduler owned by the
// factory. The 'config.num_batch_threads' threads are shared across all session
// instances created by this factory. However, each session has its own
// dedicated queue of size 'config.max_enqueued_batches'.
//
// The factory can also estimate the resource (e.g. RAM) requirements of a
// SavedModelBundle based on the SavedModel (i.e. prior to loading the session).
//
// This class is thread-safe.
class SavedModelBundleFactory {
 public:
  static Status Create(const SessionBundleConfig& config,
                       std::unique_ptr<SavedModelBundleFactory>* factory);

  // Instantiates a bundle from a given export or SavedModel path.
  Status CreateSavedModelBundle(const string& path,
                                std::unique_ptr<SavedModelBundle>* bundle);

  // Estimates the resources a SavedModel bundle will use once loaded, from its
  // export path.
  // TODO(b/33078719): remove this method after we switch all the callers to
  // the following one.
  Status EstimateResourceRequirement(const string& path,
                                     ResourceAllocation* estimate) const;

 private:
  using Batcher = SharedBatchScheduler<BatchingSessionTask>;

  SavedModelBundleFactory(const SessionBundleConfig& config,
                          std::shared_ptr<Batcher> batch_scheduler);

  const SessionBundleConfig config_;

  // A shared batch scheduler. One queue is used for each session this factory
  // emits. If batching is not configured, this remains null.
  std::shared_ptr<Batcher> batch_scheduler_;

  TF_DISALLOW_COPY_AND_ASSIGN(SavedModelBundleFactory);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_BUNDLE_FACTORY_H_
