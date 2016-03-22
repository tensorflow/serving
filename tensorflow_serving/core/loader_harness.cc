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

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {

LoaderHarness::LoaderHarness(const ServableId& id,
                             std::unique_ptr<Loader> loader)
    : LoaderHarness(id, std::move(loader), Options()) {}

LoaderHarness::LoaderHarness(const ServableId& id,
                             std::unique_ptr<Loader> loader,
                             const Options& options)
    : id_(id), loader_(std::move(loader)), options_(options) {
  VLOG(1) << "New aspired servable version " << id_;
}

LoaderHarness::~LoaderHarness() {
  mutex_lock l(mu_);
  DCHECK(state_ == kNew || state_ == kDisabled || state_ == kError) << state_;
}

LoaderHarness::State LoaderHarness::state() const {
  mutex_lock l(mu_);
  return state_;
}

ServableStateSnapshot LoaderHarness::loader_state_snapshot() const {
  mutex_lock l(mu_);
  return {id_, state_, is_aspired_};
}

Status LoaderHarness::Load(const ResourceAllocation& available_resources) {
  {
    mutex_lock l(mu_);
    DCHECK_EQ(kNew, state_);
    state_ = kLoading;
    VLOG(1) << "Loading servable version " << id_;
  }

  const Status status = [&]() {
    Status load_status;
    int num_tries = 0;
    do {
      if (num_tries > 0) {
        if (options_.load_retry_interval_micros > 0) {
          Env::Default()->SleepForMicroseconds(
              options_.load_retry_interval_micros);
        }
        LOG(INFO) << "Retrying load on servable version: " << id_
                  << " retry: " << num_tries;
      }
      load_status = loader_->Load(available_resources);
      ++num_tries;
    } while (is_aspired() && !load_status.ok() &&
             num_tries < options_.max_num_load_tries);
    return load_status;
  }();

  {
    mutex_lock l(mu_);
    DCHECK_EQ(kLoading, state_);
    if (status.ok()) {
      state_ = kReady;
      VLOG(1) << "Successfully loaded servable version " << id_;
    } else {
      ErrorInternal(status);
    }
  }
  return status;
}

void LoaderHarness::Unload() {
  {
    mutex_lock l(mu_);
    DCHECK_EQ(state_, kQuiesced);
    state_ = kUnloading;
    VLOG(1) << "Unloading servable version " << id_;
  }
  loader_->Unload();
  {
    mutex_lock l(mu_);
    DCHECK_EQ(state_, kUnloading);
    state_ = kDisabled;
    VLOG(1) << "Done unloading servable version " << id_;
  }
}

bool LoaderHarness::is_aspired() const {
  mutex_lock l(mu_);
  return is_aspired_;
}

void LoaderHarness::set_is_aspired(const bool is_aspired) {
  mutex_lock l(mu_);
  // Only log if the value actually changes.
  if (is_aspired != is_aspired_) {
    is_aspired_ = is_aspired;
    VLOG(1) << "Setting servable version " << id_ << " as "
            << (is_aspired ? "" : "not ") << "aspired";
  }
}

void LoaderHarness::StartQuiescing() {
  mutex_lock l(mu_);
  DCHECK_EQ(state_, kReady);
  state_ = kQuiescing;
  VLOG(1) << "Quiescing servable version " << id_;
}

void LoaderHarness::DoneQuiescing() {
  mutex_lock l(mu_);
  DCHECK_EQ(state_, kQuiescing);
  state_ = kQuiesced;
  VLOG(1) << "Done quiescing servable version " << id_;
}

void LoaderHarness::ErrorInternal(const Status status) {
  state_ = kError;
  status_ = status;
  VLOG(1) << "Encountered an error for servable version " << id_ << ": "
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

}  // namespace serving
}  // namespace tensorflow
