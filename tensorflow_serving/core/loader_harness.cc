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
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {

LoaderHarness::LoaderHarness(const ServableId& id,
                             std::unique_ptr<Loader> loader,
                             const Options& options)
    : id_(id),
      loader_(std::move(loader)),
      additional_state_(nullptr),
      options_(options) {}

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
    return errors::FailedPrecondition(
        "Servable: ", id_.DebugString(),
        " cannot be transitioned to load-requested. In state: ",
        StateDebugString(state_), " instead of state: ",
        StateDebugString(State::kNew));
  }
  state_ = State::kLoadRequested;
  return Status::OK();
}

Status LoaderHarness::LoadApproved() {
  mutex_lock l(mu_);

  if (state_ != State::kLoadRequested) {
    return errors::FailedPrecondition(
        "Servable: ", id_.DebugString(),
        " cannot be approved for loading. In state: ", StateDebugString(state_),
        " instead of state: ", StateDebugString(State::kLoadRequested));
  }
  state_ = State::kLoadApproved;
  LOG(INFO) << "Approving load for servable version " << id_;

  return Status::OK();
}

Status LoaderHarness::Load(const ResourceAllocation& available_resources) {
  {
    mutex_lock l(mu_);
    if (state_ != State::kLoadApproved) {
      return errors::FailedPrecondition(
          "Servable: ", id_.DebugString(), " cannot be loaded. In state: ",
          StateDebugString(state_), " instead of state: ",
          StateDebugString(State::kLoadApproved));
    }
    state_ = State::kLoading;
    LOG(INFO) << "Loading servable version " << id_;
  }

  const Status status = [&]() {
    Status load_status;
    int num_tries = 0;
    do {
      if (num_tries > 0) {
        if (cancel_load_retry()) {
          LOG(INFO) << "Load retry cancelled for servable: " << id_;
          break;
        }
        Env::Default()->SleepForMicroseconds(
            options_.load_retry_interval_micros);
        LOG(INFO) << "Retrying load on servable version: " << id_
                  << " retry: " << num_tries;
      }
      load_status = loader_->Load(available_resources);
      if (!load_status.ok()) {
        LOG(ERROR) << "Servable: " << id_ << " load failure: " << load_status;
      }
      ++num_tries;
    } while (!cancel_load_retry() && !load_status.ok() &&
             (num_tries - 1) < options_.max_num_load_retries);

    return load_status;
  }();

  {
    mutex_lock l(mu_);
    DCHECK_EQ(State::kLoading, state_);
    if (status.ok()) {
      state_ = State::kReady;
      LOG(INFO) << "Successfully loaded servable version " << id_;
    } else {
      ErrorInternal(status);
    }
  }

  return status;
}

Status LoaderHarness::UnloadRequested() {
  mutex_lock l(mu_);

  if (state_ != State::kReady) {
    return errors::FailedPrecondition(
        "Servable: ", id_.DebugString(),
        " cannot be transitioned to unload-requested. In state: ",
        StateDebugString(state_), " instead of state: ",
        StateDebugString(State::kReady));
  }
  state_ = State::kUnloadRequested;
  return Status::OK();
}

void LoaderHarness::set_cancel_load_retry(const bool value) {
  mutex_lock l(mu_);
  cancel_load_retry_ = value;
}

bool LoaderHarness::cancel_load_retry() {
  mutex_lock l(mu_);
  return cancel_load_retry_;
}

void LoaderHarness::Unload() {
  {
    mutex_lock l(mu_);
    DCHECK_EQ(state_, State::kQuiesced);
    state_ = State::kUnloading;
    LOG(INFO) << "Unloading servable version " << id_;
  }

  loader_->Unload();

  {
    mutex_lock l(mu_);
    DCHECK_EQ(state_, State::kUnloading);
    state_ = State::kDisabled;
    LOG(INFO) << "Done unloading servable version " << id_;
  }
}

Status LoaderHarness::StartQuiescing() {
  mutex_lock l(mu_);
  if (state_ != State::kUnloadRequested) {
    return errors::FailedPrecondition(
        "Servable: ", id_.DebugString(), " cannot be quiesced. In state: ",
        StateDebugString(state_), " instead of state: ",
        StateDebugString(State::kUnloadRequested));
  }
  state_ = State::kQuiescing;
  LOG(INFO) << "Quiescing servable version " << id_;
  return Status::OK();
}

void LoaderHarness::DoneQuiescing() {
  mutex_lock l(mu_);
  DCHECK_EQ(state_, State::kQuiescing);
  state_ = State::kQuiesced;
  LOG(INFO) << "Done quiescing servable version " << id_;
}

void LoaderHarness::ErrorInternal(const Status status) {
  state_ = State::kError;
  status_ = status;
  LOG(INFO) << "Encountered an error for servable version " << id_ << ": "
            << status_;
}

void LoaderHarness::Error(const Status status) {
  mutex_lock l(mu_);
  ErrorInternal(status);
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
