/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/core/prefix_storage_path_source_adapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/storage_path.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(PrefixStoragePathSourceAdapterTest, Basic) {
  {
    PrefixStoragePathSourceAdapter adapter("");
    ServableData<StoragePath> output =
        adapter.AdaptOneVersion(ServableData<StoragePath>({"foo", 42}, "bar"));
    EXPECT_EQ("foo", output.id().name);
    EXPECT_EQ(42, output.id().version);
    EXPECT_EQ("bar", output.DataOrDie());
  }

  {
    PrefixStoragePathSourceAdapter adapter("/baz");
    ServableData<StoragePath> output =
        adapter.AdaptOneVersion(ServableData<StoragePath>({"foo", 42}, "bar"));
    EXPECT_EQ("foo", output.id().name);
    EXPECT_EQ(42, output.id().version);
    EXPECT_EQ("/baz/bar", output.DataOrDie());
  }

  {
    PrefixStoragePathSourceAdapter adapter("/baz/");
    ServableData<StoragePath> output =
        adapter.AdaptOneVersion(ServableData<StoragePath>({"foo", 42}, "bar"));
    EXPECT_EQ("foo", output.id().name);
    EXPECT_EQ(42, output.id().version);
    EXPECT_EQ("/baz/bar", output.DataOrDie());
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
