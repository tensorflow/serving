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
#include "tensorflow_serving/util/class_registration.h"

namespace tensorflow {
namespace serving {

// Factory for returning intra- and inter-op thread pools to be used by
// Tensorflow.
class ThreadPoolFactory {
 public:
  virtual ~ThreadPoolFactory() = default;
  virtual thread::ThreadPoolInterface* GetInterOpThreadPool() = 0;
  virtual thread::ThreadPoolInterface* GetIntraOpThreadPool() = 0;
};

DEFINE_CLASS_REGISTRY(ThreadPoolFactoryRegistry, ThreadPoolFactory);
#define REGISTER_THREAD_POOL_FACTORY(ClassCreator, ConfigProto)              \
  REGISTER_CLASS(ThreadPoolFactoryRegistry, ThreadPoolFactory, ClassCreator, \
                 ConfigProto);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_THREAD_POOL_FACTORY_H_
