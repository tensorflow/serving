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

#include "tensorflow_serving/core/loader_harness.h"

#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/util/retrier.h"

namespace tensorflow {
namespace serving {

LoaderHarness::LoaderHarness(const ServableId& id,
                             std::unique_ptr<Loader> loader,
                             const Options& options)
    : id_(id),
      loader_(std::move(loader)),
      additional_state_(nullptr),
      options_(options) {
  VLOG(1) << "Starting to manage servable version " << id_;
}

LoaderHarness::~LoaderHarness() {
  mutex_lock l(mu_);
  DCHECK(state_ == State::kNew || state_ == State::kDisabled ||
         state_ == State::kError)
      << "Servable: " << id_ << " state: " << state_;
}

LoaderHarness::State LoaderHarness::state() const {
  mutex_lock l(mu_);
  return state_;
}

Status LoaderHarness::LoadRequested() {
  mutex_lock l(mu_);

  if (state_ != State::kNew) {
    return errors::FailedPrecondition("Duplicate load request");
  }
  state_ = State::kLoadRequested;
  VLOG(1) << "Load requested for servable version " << id_;

  return OkStatus();
}

Status LoaderHarness::LoadApproved() {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(
      TransitionState(State::kLoadRequested, State::kLoadApproved));
  LOG(INFO) << "Approving load for servable version " << id_;
  return OkStatus();
}

Status LoaderHarness::Load() {
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(TransitionState(State::kLoadApproved, State::kLoading));
    LOG(INFO) << "Loading servable version " << id_;
  }

  const Status status = Retry(
      strings::StrCat("Loading servable: ", id_.DebugString()),
      options_.max_num_load_retries, options_.load_retry_interval_micros,
      [&]() { return loader_->LoadWithMetadata({id_}); },
      [&]() { return cancel_load_retry(); });

  if (status.ok()) {
    if (cancel_load_retry()) {
      // Servable is going to be unloaded very soon,
      // we report a failure here so that we do not accidentally
      // report that the servable is available.
      TF_RETURN_IF_ERROR(UnloadDueToCancelledLoad());
      return errors::Cancelled(
          strings::StrCat("Loading of servable cancelled"));
    }
    {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(TransitionState(State::kLoading, State::kReady));
      LOG(INFO) << "Successfully loaded servable version " << id_;
    }
  } else {
    mutex_lock l(mu_);
    ErrorInternal(status);
  }

  return status;
}

Status LoaderHarness::UnloadRequested() {
  mutex_lock l(mu_);
  if (state_ != State::kReady) {
    return errors::FailedPrecondition(
        "Servable not loaded, or unload already requested/ongoing");
  }
  state_ = State::kUnloadRequested;
  return OkStatus();
}

Status LoaderHarness::UnloadInternal(State from_state) {
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(TransitionState(from_state, State::kUnloading));
    LOG(INFO) << "Unloading just-loaded servable version " << id_;
  }

  loader_->Unload();

  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(TransitionState(State::kUnloading, State::kDisabled));
    LOG(INFO) << "Done unloading servable version " << id_;
  }
  return OkStatus();
}

Status LoaderHarness::UnloadDueToCancelledLoad() {
  return UnloadInternal(State::kLoading);
}

void LoaderHarness::set_cancel_load_retry(const bool value) {
  mutex_lock l(mu_);
  cancel_load_retry_ = value;
}

bool LoaderHarness::cancel_load_retry() {
  mutex_lock l(mu_);
  return cancel_load_retry_;
}

Status LoaderHarness::Unload() { return UnloadInternal(State::kQuiesced); }

Status LoaderHarness::StartQuiescing() {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(
      TransitionState(State::kUnloadRequested, State::kQuiescing));
  LOG(INFO) << "Quiescing servable version " << id_;
  return OkStatus();
}

Status LoaderHarness::DoneQuiescing() {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(TransitionState(State::kQuiescing, State::kQuiesced));
  LOG(INFO) << "Done quiescing servable version " << id_;
  return OkStatus();
}

void LoaderHarness::ErrorInternal(const Status& status) {
  state_ = State::kError;
  status_ = status;
  if (options_.error_callback) {
    options_.error_callback(id(), status);
  }
  LOG(INFO) << "Encountered an error for servable version " << id_ << ": "
            << status_;
}

void LoaderHarness::Error(const Status& status) {
  mutex_lock l(mu_);
  ErrorInternal(status);
}

Status LoaderHarness::TransitionState(const State from, const State to) {
  if (state_ != from) {
    const Status error = errors::Internal(
        "Illegal request to transition from state ", StateDebugString(state_),
        " to ", StateDebugString(to));
#ifndef NDEBUG
    LOG(FATAL) << error;  // Crash OK
#else
    ErrorInternal(error);
#endif
    return error;
  }
  state_ = to;
  return OkStatus();
}

Status LoaderHarness::status() const {
  mutex_lock l(mu_);
  return status_;
}

string LoaderHarness::StateDebugString(const State state) {
  switch (state) {
    case State::kNew:
      return "new";
    case State::kLoadRequested:
      return "load-requested";
    case State::kLoadApproved:
      return "load-approved";
    case State::kLoading:
      return "loading";
    case State::kReady:
      return "ready";
    case State::kUnloadRequested:
      return "unload-requested";
    case State::kQuiescing:
      return "quiescing";
    case State::kQuiesced:
      return "quiesced";
    case State::kUnloading:
      return "unloading";
    case State::kDisabled:
      return "disabled";
    case State::kError:
      return "error";
  }
}

}  // namespace serving
}  // namespace tensorflow
