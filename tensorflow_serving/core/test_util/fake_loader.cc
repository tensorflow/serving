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

#include "tensorflow_serving/core/test_util/fake_loader.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace serving {
namespace test_util {

thread_local bool FakeLoader::was_deleted_in_this_thread_;
int FakeLoader::num_fake_loaders_ = 0;
mutex FakeLoader::num_fake_loaders_mu_(LINKER_INITIALIZED);

FakeLoader::FakeLoader(int64 servable, const Status load_status)
    : servable_(servable), load_status_(load_status) {
  was_deleted_in_this_thread_ = false;
  {
    mutex_lock l(num_fake_loaders_mu_);
    ++num_fake_loaders_;
  }
}

FakeLoader::~FakeLoader() {
  {
    mutex_lock l(num_fake_loaders_mu_);
    --num_fake_loaders_;
  }
  was_deleted_in_this_thread_ = true;
}

Status FakeLoader::load_status() { return load_status_; }

Status FakeLoader::Load() { return load_status_; }

void FakeLoader::Unload() {}

AnyPtr FakeLoader::servable() { return AnyPtr(&servable_); }

bool FakeLoader::was_deleted_in_this_thread() {
  return was_deleted_in_this_thread_;
}

int FakeLoader::num_fake_loaders() {
  mutex_lock l(num_fake_loaders_mu_);
  return num_fake_loaders_;
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
