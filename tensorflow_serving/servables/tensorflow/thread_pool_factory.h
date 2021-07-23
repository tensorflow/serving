/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_THREAD_POOL_FACTORY_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_THREAD_POOL_FACTORY_H_

#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/util/class_registration.h"

namespace tensorflow {
namespace serving {

// This class takes inter- and intra-op thread pools and returns
// tensorflow::thread::ThreadPoolOptions. The thread pools passed to an
// instance of this class will be kept alive for the lifetime of this instance.
class ScopedThreadPools {
 public:
  // The default constructor will set inter- and intra-op thread pools in the
  // ThreadPoolOptions to nullptr, which will be ingored by Tensorflow runtime.
  ScopedThreadPools() = default;
  ScopedThreadPools(
      std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool,
      std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool);
  ~ScopedThreadPools() = default;

  tensorflow::thread::ThreadPoolOptions get();

 private:
  std::shared_ptr<thread::ThreadPoolInterface> inter_op_thread_pool_;
  std::shared_ptr<thread::ThreadPoolInterface> intra_op_thread_pool_;
};

// Factory for returning intra- and inter-op thread pools to be used by
// Tensorflow.
class ThreadPoolFactory {
 public:
  virtual ~ThreadPoolFactory() = default;

  virtual ScopedThreadPools GetThreadPools() = 0;
};

DEFINE_CLASS_REGISTRY(ThreadPoolFactoryRegistry, ThreadPoolFactory);
#define REGISTER_THREAD_POOL_FACTORY(ClassCreator, ConfigProto)              \
  REGISTER_CLASS(ThreadPoolFactoryRegistry, ThreadPoolFactory, ClassCreator, \
                 ConfigProto);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_THREAD_POOL_FACTORY_H_
