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

#ifndef TENSORFLOW_SERVING_CORE_LOADER_HARNESS_H_
#define TENSORFLOW_SERVING_CORE_LOADER_HARNESS_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

// Forward-declaration for the use in LoaderHarness.
struct ServableStateSnapshot;

// LoaderHarness is a widget the Manager uses to hold on to and talk
// to a Loader while it owns it. It tracks the overall state of a Servable
// such that Manager can determine which state transitions to make at
// what times. LoaderHarness is thread-safe.
class LoaderHarness final {
 public:
  // State of the underlying servable, from the perspective of the LoaderHarness
  // and for the purpose of communication between it and a Manager. Not
  // equivalent to the semantic servable states in servable_state.h.
  //
  // Valid transitions:
  // kNew-->kLoading-->kReady-->kQuiescing-->kQuiesced-->kUnloading-->kDisabled
  // as well as: any_state-->kError.
  enum State {
    // Initial state.
    kNew = 0,

    // 'loader_->Load()' has been called.
    kLoading,

    // 'loader_->Load()' has succeeded.
    kReady,

    // The servable is going to be made unavailable for serving.
    kQuiescing,

    // The servable has been made unavailable for serving.
    kQuiesced,

    // 'loader_->Unload()' has been called.
    kUnloading,

    // 'loader_->Unload()' has finished.
    kDisabled,

    // An error has occurred, either during 'loader_->Load()' or outside of the
    // harness (and was reported to the harness via a call to Error()).
    kError
  };

  struct Options {
    // Maximum number of times we try to load a servable before we give up.
    int max_num_load_tries;

    // The interval, in microseconds, between each servable load retry.
    int64 load_retry_interval_micros;
  };

  LoaderHarness(const ServableId& id, std::unique_ptr<Loader> loader);
  LoaderHarness(const ServableId& id, std::unique_ptr<Loader> loader,
                const Options& options);

  // Legal to destruct iff current state is kNew|kDisabled|kError.
  // Check-fails if violated.
  ~LoaderHarness();

  // Returns the identifier of underlying Servable.
  ServableId id() const { return id_; }

  // Returns the current state of underlying Servable.
  State state() const LOCKS_EXCLUDED(mu_);

  // Returns a pointer to the wrapped loader.
  // Ownership remains with this class.
  Loader* loader() const { return loader_.get(); }

  // Returns the current overall state snapshot of the underlying Servable.
  ServableStateSnapshot loader_state_snapshot() const LOCKS_EXCLUDED(mu_);

  // Transitions to kLoading, delegates to Servable::Load(), then transitions
  // either to kReady if Load() succeeds, or to kError if Load() fails. This
  // call may take a long time.
  //
  // We retry the Servable::Load() according to the options set during
  // construction of this class. We stop retrying and give up if 1. we have
  // reached max_num_load_tries or, 2. if is_aspired is set to false.
  //
  // Legal to call iff current state is kNew. Check-fails if violated.
  Status Load(const ResourceAllocation& available_resources)
      LOCKS_EXCLUDED(mu_);

  // Transitions to kUnloading, delegates to Servable::Unload(), then
  // transitions to kDisabled when Unload() is done.
  void Unload() LOCKS_EXCLUDED(mu_);

  // Returns whether the underlying Servable is *aspired*. An aspired
  // Servable should be loaded to serve user requests if resources
  // permit. The underlying Servable is always set to aspired upon
  // creation of LoaderHarness. Once it is no longer the case, it
  // should eventually be unloaded to free up resources.
  bool is_aspired() const LOCKS_EXCLUDED(mu_);

  // Sets whether the underlying Servable is aspired.
  // Note that this method simply remembers the aspired-state. It is the
  // responsibility of ServableManager to eventually drive the
  // state transition.
  void set_is_aspired(bool is_aspired) LOCKS_EXCLUDED(mu_);

  // Transitions the state to kQuiescing, which means that we would like to not
  // give out any more handles to this servable.
  void StartQuiescing() LOCKS_EXCLUDED(mu_);

  // Transitions the state to kQuiesced, which means that there are no more live
  // handles to this servable available in the frontend. At this point, we can
  // unload this object.
  void DoneQuiescing() LOCKS_EXCLUDED(mu_);

  // Transitions the state to kError.
  void Error(Status status) LOCKS_EXCLUDED(mu_);

  // Whether anything has gone wrong with this servable. If state is kError,
  // this will be non-OK. If not OK, the error could be something that occurred
  // in a Source or SourceAdapter, in the Loader, in the Manager, or elsewhere.
  // All errors pertaining to the servable are reported here, regardless of
  // origin.
  Status status() const LOCKS_EXCLUDED(mu_);

 private:
  // Transitions the state to kError. Private method to be used when we want to
  // set an error from another method in this class, where mu_ is already held.
  void ErrorInternal(Status status) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const ServableId id_;
  const std::unique_ptr<Loader> loader_;
  const Options options_;
  mutable mutex mu_;
  State state_ GUARDED_BY(mu_) = kNew;
  bool is_aspired_ GUARDED_BY(mu_) = true;
  // If state_ is kError, this will be non-OK.
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(LoaderHarness);
};

// A snapshot of a servable's state and aspiredness, from the LoaderHarness's
// perspective.
struct ServableStateSnapshot final {
  ServableId id;
  LoaderHarness::State state;
  bool is_aspired;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_LOADER_HARNESS_H_
