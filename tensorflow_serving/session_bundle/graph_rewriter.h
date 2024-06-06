/* Copyright 2019 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SESSION_BUNDLE_GRAPH_REWRITER_H_
#define TENSORFLOW_SERVING_SESSION_BUNDLE_GRAPH_REWRITER_H_

#include <functional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace serving {

// Class for registering a global graph rewrite function.
class GraphRewriter {
 public:
  static GraphRewriter& GetGlobal() {
    static auto* const singleton = new GraphRewriter();
    return *singleton;
  }

  Status Set(std::function<Status(tensorflow::MetaGraphDef*)>&& rewriter) {
    absl::MutexLock l(&m_);
    if (rewriter_ != nullptr)
      return errors::AlreadyExists("Graph rewriter already set.");

    rewriter_ = std::move(rewriter);

    return Status();
  }

  // For testing only. Resets the rewriter to nullptr
  Status ResetForTesting() {
    absl::MutexLock l(&m_);
    rewriter_ = nullptr;
    return Status();
  }

  std::function<Status(tensorflow::MetaGraphDef*)>& Get() {
    absl::MutexLock l(&m_);
    return rewriter_;
  }

  bool IsRegistered() {
    absl::MutexLock l(&m_);
    return rewriter_ != nullptr;
  }

 private:
  GraphRewriter() = default;

  absl::Mutex m_;
  std::function<Status(tensorflow::MetaGraphDef*)> rewriter_
      ABSL_GUARDED_BY(m_);
};

// EXPERIMENTAL. THE 2 METHODS BELOW MAY CHANGE OR GO AWAY. USE WITH CAUTION.
// Sets a global graph rewrite function that is called on all saved models
// immediately after metagraph load, but before session creation.  This function
// can only be called once.
inline Status SetGraphRewriter(
    std::function<Status(tensorflow::MetaGraphDef*)>&& rewriter) {
  return GraphRewriter::GetGlobal().Set(std::move(rewriter));
}

// For testing only. Resets the experimental graph rewriter above.
inline Status ResetGraphRewriterForTesting() {
  return GraphRewriter::GetGlobal().ResetForTesting();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SESSION_BUNDLE_GRAPH_REWRITER_H_
