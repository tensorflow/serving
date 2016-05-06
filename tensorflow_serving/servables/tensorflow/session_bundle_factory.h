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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_FACTORY_H_

#include "tensorflow_serving/batching/batching_session.h"
#include "tensorflow_serving/batching/shared_batch_scheduler.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"

namespace tensorflow {
namespace serving {

// A factory that creates SessionBundles from export paths.
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
// SessionBundle based on the export (i.e. prior to loading the session).
//
// This class is thread-safe.
class SessionBundleFactory {
 public:
  // Constants used in the resource estimation heuristic. See the documentation
  // on EstimateResourceRequirements().
  static constexpr double kResourceEstimateRAMMultiplier = 1.5;
  static constexpr int kResourceEstimateRAMPadBytes = 10 << 20 /* 10 MB */;

  static Status Create(const SessionBundleConfig& config,
                       std::unique_ptr<SessionBundleFactory>* factory);

  // Instantiates a bundle from a given export path.
  Status CreateSessionBundle(const string& path,
                             std::unique_ptr<SessionBundle>* bundle);

  // Estimates the resources a session bundle will use once loaded, from its
  // export path.
  //
  // Uses the following crude heuristic, for now: estimated main-memory RAM =
  // (combined size of variable file(s)) * kResourceEstimateRAMMultiplier +
  // kResourceEstimateRAMPadBytes.
  // TODO(b/27694447): Improve the heuristic. At a minimum, account for GPU RAM.
  Status EstimateResourceRequirement(const string& path,
                                     ResourceAllocation* estimate) const;

 private:
  using Batcher = SharedBatchScheduler<BatchingSessionTask>;

  SessionBundleFactory(const SessionBundleConfig& config,
                       std::shared_ptr<Batcher> batch_scheduler);

  // Wraps a batching queue with retrier around 'bundle->session'.
  Status WrapSessionForBatching(SessionBundle* bundle);

  const SessionBundleConfig config_;

  // A shared batch scheduler. One queue is used for each session this adapter
  // emits. If batching is not configured, this remains null.
  std::shared_ptr<Batcher> batch_scheduler_;

  TF_DISALLOW_COPY_AND_ASSIGN(SessionBundleFactory);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SESSION_BUNDLE_FACTORY_H_
