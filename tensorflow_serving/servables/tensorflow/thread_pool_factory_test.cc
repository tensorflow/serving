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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(ScopedThreadPools, DefaultCtor) {
  ScopedThreadPools thread_pools;
  EXPECT_EQ(nullptr, thread_pools.get().inter_op_threadpool);
  EXPECT_EQ(nullptr, thread_pools.get().intra_op_threadpool);
}

TEST(ScopedThreadPools, NonDefaultCtor) {
  auto inter_op_thread_pool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "InterOp",
                                                      /*num_threads=*/1);
  auto intra_op_thread_pool =
      std::make_shared<test_util::CountingThreadPool>(Env::Default(), "InterOp",
                                                      /*num_threads=*/1);
  ScopedThreadPools thread_pools(inter_op_thread_pool, intra_op_thread_pool);
  EXPECT_EQ(inter_op_thread_pool.get(), thread_pools.get().inter_op_threadpool);
  EXPECT_EQ(intra_op_thread_pool.get(), thread_pools.get().intra_op_threadpool);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
