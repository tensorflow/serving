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
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// Forward-declaration for the use in LoaderHarness.
template <typename T>
struct ServableStateSnapshot;

// LoaderHarness is a widget the Manager uses to hold on to and talk to a Loader
// while it owns it. It tracks the overall state of a Servable such that Manager
// can determine which state transitions to make at what times.
//
// A manager implementation can also add some additional state with each
// harness. For example, a manager could put ACL or lifecycle metadata here. The
// ownership is maintained by the harness.
//
// This class is thread-safe.
class LoaderHarness final {
 public:
  // State of the underlying servable, from the perspective of the LoaderHarness
  // and for the purpose of communication between it and a Manager. Not
  // equivalent to the semantic servable states in servable_state.h.
  //
  // Valid transitions:
  // kNew-->kLoading-->kReady-->kQuiescing-->kQuiesced-->kUnloading-->kDisabled
  // as well as: any_state-->kError.
  enum class State {
    // Initial state.
    kNew,

    // The manager has been requested to load this servable.
    kLoadRequested,

    // The servable has been approved for loading, e.g. resources have been set
    // aside for it.
    kLoadApproved,

    // 'loader_->Load()' has been called.
    kLoading,

    // 'loader_->Load()' has succeeded.
    kReady,

    // The manager has been requested to unload this servable.
    kUnloadRequested,

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
    Options() {}

    // Maximum number of times we retry loading a servable, after the first
    // failure, before we give up.
    uint32 max_num_load_retries = 0;

    // The interval, in microseconds, between each servable load retry.
    uint64 load_retry_interval_micros = 0;

    // An (optional) function to call upon transitioning to state kError.
    std::function<void(const ServableId& id, const Status& error)>
        error_callback;
  };

  LoaderHarness(const ServableId& id, std::unique_ptr<Loader> loader,
                const Options& options = Options());

  // Constructor to create a harness with additional state, which a manager
  // needs.
  template <typename T>
  LoaderHarness(const ServableId& id, std::unique_ptr<Loader> loader,
                std::unique_ptr<T> additional_state,
                const Options& options = Options())
      : id_(id),
        loader_(std::move(loader)),
        additional_state_(std::move(additional_state)),
        options_(options) {}

  // Legal to destruct iff current state is one of kNew, kDisabled or kError.
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
  template <typename T = std::nullptr_t>
  ServableStateSnapshot<T> loader_state_snapshot() const LOCKS_EXCLUDED(mu_);

  // Transitions the state of the harness to kLoadRequested iff its current
  // state is kNew. The test-and-change is done transactionally, so this method
  // can be used to ensure that at most one Load() request can proceed.
  Status LoadRequested() LOCKS_EXCLUDED(mu_);

  // Transitions to kLoadApproved.
  //
  // REQUIRES: State is kLoadRequested when called. Otherwise DCHECK-fails,
  // transitions to state kError and invokes 'options_.error_callback'.
  Status LoadApproved() LOCKS_EXCLUDED(mu_);

  // Transitions to kLoading, delegates to Servable::Load(), then transitions
  // either to kReady if Load() succeeds, or to kError (and invokes 'options_.
  // error_callback') if Load() fails. This call may take a long time.
  //
  // We retry the Servable::Load() according to the options set during
  // construction of this class. We stop retrying and give up if 1. we have
  // reached max_num_load_retries or, 2. if cancel_load() is set to true.
  //
  // REQUIRES: State is kLoadApproved when called. Otherwise DCHECK-fails,
  // transitions to state kError and invokes 'options_.error_callback'.
  Status Load() LOCKS_EXCLUDED(mu_);

  // Transitions the state of the harness to kUnloadRequested iff its current
  // state is kReady. The test-and-change is done transactionally, so this
  // method can be used to ensure that at most one Load() request can proceed.
  Status UnloadRequested() LOCKS_EXCLUDED(mu_);

  // Cancels retrying the load of the servable. This is best-effort, and does
  // not preempt a Load() which is already happening, only subsequent calls.
  //
  // If the retries are cancelled, the servable goes into a state dependent on
  // the last Load() called on it. If the last Load() was successful, it will be
  // in state kReady, else in kError.
  void set_cancel_load_retry(bool value) LOCKS_EXCLUDED(mu_);
  bool cancel_load_retry() LOCKS_EXCLUDED(mu_);

  // Transitions to kUnloading, delegates to Servable::Unload(), then
  // transitions to kDisabled when Unload() is done.
  //
  // REQUIRES: State is kQuiesced when called. Otherwise DCHECK-fails,
  // transitions to state kError and invokes 'options_.error_callback'.
  Status Unload() LOCKS_EXCLUDED(mu_);

  // Transitions the state to kQuiescing, which means that we would like to not
  // give out any more handles to this servable.
  //
  // REQUIRES: State is kUnloadRequested when called. Otherwise DCHECK-fails,
  // transitions to state kError and invokes 'options_.error_callback'.
  Status StartQuiescing() LOCKS_EXCLUDED(mu_);

  // Transitions the state to kQuiesced, which means that there are no more live
  // handles to this servable available in the frontend. At this point, we can
  // unload this object.
  //
  // REQUIRES: State is kQuiescing when called. Otherwise DCHECK-fails,
  // transitions to state kError and invokes 'options_.error_callback'.
  Status DoneQuiescing() LOCKS_EXCLUDED(mu_);

  // Transitions the state to kError and invokes 'options_.error_callback'.
  void Error(const Status& status) LOCKS_EXCLUDED(mu_);

  // Whether anything has gone wrong with this servable. If state is kError,
  // this will be non-OK. If not OK, the error could be something that occurred
  // in a Source or SourceAdapter, in the Loader, in the Manager, or elsewhere.
  // All errors pertaining to the servable are reported here, regardless of
  // origin.
  Status status() const LOCKS_EXCLUDED(mu_);

  // Gets the additional state. Returns nullptr if the type mismatches or if it
  // wasn't set.
  template <typename T>
  T* additional_state() {
    return additional_state_.get<T>();
  }

  static string StateDebugString(State state);

 private:
  // Transitions the state to kError and invokes 'options_.error_callback'.
  // Private method to be used when we want to set an error from another method
  // in this class, where mu_ is already held.
  void ErrorInternal(const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Expects 'state_' to equal 'from', and if so transitions it to 'to'. If not,
  // DCHECK-fails, calls ErrorInternal() with a suitable error and returns the
  // same error.
  Status TransitionState(State from, State to) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const ServableId id_;
  const std::unique_ptr<Loader> loader_;
  // Additional state that the manager uses.
  const UniqueAnyPtr additional_state_;
  const Options options_;
  mutable mutex mu_;
  State state_ GUARDED_BY(mu_) = State::kNew;
  // If state_ is kError, this will be non-OK.
  Status status_ GUARDED_BY(mu_);
  // If set to true, we don't try to retry the load of the servable, if not
  // loaded by the first attempt.
  bool cancel_load_retry_ GUARDED_BY(mu_) = false;

  TF_DISALLOW_COPY_AND_ASSIGN(LoaderHarness);
};

// A snapshot of a servable's state and aspiredness, from the LoaderHarness's
// perspective.
template <typename T = std::nullptr_t>
struct ServableStateSnapshot final {
  ServableId id;
  LoaderHarness::State state;
  optional<T> additional_state;
};

template <typename T>
inline bool operator==(const ServableStateSnapshot<T>& a,
                       const ServableStateSnapshot<T>& b) {
  return a.id == b.id && a.state == b.state &&
         a.additional_state == b.additional_state;
}

template <typename T>
inline bool operator!=(const ServableStateSnapshot<T>& a,
                       const ServableStateSnapshot<T>& b) {
  return !(a == b);
}

inline std::ostream& operator<<(std::ostream& os, LoaderHarness::State state) {
  os << LoaderHarness::StateDebugString(state);
  return os;
}

////
// Implementation details. API readers may skip.
////

template <typename T>
ServableStateSnapshot<T> LoaderHarness::loader_state_snapshot() const {
  mutex_lock l(mu_);
  if (additional_state_.get<T>() == nullptr) {
    return {id_, state_, {}};
  } else {
    return {id_, state_, {*additional_state_.get<T>()}};
  }
}

// Specialization for std::nullptr_t.
//
// We mark this inline to follow ODR.
template <>
inline ServableStateSnapshot<std::nullptr_t>
LoaderHarness::loader_state_snapshot() const {
  mutex_lock l(mu_);
  return {id_, state_, {}};
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_LOADER_HARNESS_H_
