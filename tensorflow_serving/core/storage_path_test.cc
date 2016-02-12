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

#include "tensorflow_serving/core/storage_path.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(StoragePathTest, ServableDataEquality) {
  ServableId id0 = {"0", 0};
  ServableId id1 = {"1", 1};

  ServableData<StoragePath> a(id0, "x");
  ServableData<StoragePath> a2(id0, "x");
  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a == a2);
  EXPECT_TRUE(a2 == a);

  ServableData<StoragePath> b(id0, "y");
  ServableData<StoragePath> c(id1, "x");
  ServableData<StoragePath> d(id0, errors::Unknown("error"));
  for (const ServableData<StoragePath>& other : {b, c, d}) {
    EXPECT_TRUE(other == other);
    EXPECT_FALSE(a == other);
    EXPECT_FALSE(other == a);
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
