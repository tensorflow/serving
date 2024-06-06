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

#include "tensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.h"

#include <memory>

namespace tensorflow {
namespace serving {
namespace test_util {

Status FakeThreadPoolFactory::Create(
    const FakeThreadPoolFactoryConfig& config,
    std::unique_ptr<ThreadPoolFactory>* result) {
  *result = std::make_unique<FakeThreadPoolFactory>(config);
  return Status();
}

REGISTER_THREAD_POOL_FACTORY(FakeThreadPoolFactory,
                             FakeThreadPoolFactoryConfig);

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow
