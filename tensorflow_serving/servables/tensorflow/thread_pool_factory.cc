/* Copyright 2021 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

ScopedThreadPools::ScopedThreadPools(
    std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool,
    std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool)
    : inter_op_thread_pool_(std::move(inter_op_thread_pool)),
      intra_op_thread_pool_(std::move(intra_op_thread_pool)) {}

tensorflow::thread::ThreadPoolOptions ScopedThreadPools::get() {
  tensorflow::thread::ThreadPoolOptions options;
  options.inter_op_threadpool = inter_op_thread_pool_.get();
  options.intra_op_threadpool = intra_op_thread_pool_.get();
  return options;
}

}  // namespace serving
}  // namespace tensorflow
