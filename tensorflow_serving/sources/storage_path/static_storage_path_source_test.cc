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

#include "tensorflow_serving/sources/storage_path/static_storage_path_source.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/core/test_util/mock_storage_path_target.h"
#include "tensorflow_serving/test_util/test_util.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Return;
using ::testing::StrictMock;

namespace tensorflow {
namespace serving {
namespace {

TEST(StaticStoragePathSourceTest, Basic) {
  auto config = test_util::CreateProto<StaticStoragePathSourceConfig>(
      "servable_name: 'test_servable_name' "
      "version_num: 42 "
      "version_path: 'test_version_path' ");
  std::unique_ptr<StaticStoragePathSource> source;
  TF_ASSERT_OK(StaticStoragePathSource::Create(config, &source));

  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  EXPECT_CALL(*target, SetAspiredVersions(Eq("test_servable_name"),
                                          ElementsAre(ServableData<StoragePath>(
                                              {"test_servable_name", 42},
                                              "test_version_path"))));
  ConnectSourceToTarget(source.get(), target.get());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
