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

#include "tensorflow_serving/core/dynamic_source_router.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/core/test_util/mock_storage_path_target.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::StrictMock;

namespace tensorflow {
namespace serving {
namespace {

TEST(DynamicSourceRouterTest, InvalidRouteMap) {
  std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
  // Negative output port.
  EXPECT_FALSE(
      DynamicSourceRouter<string>::Create(2, {{"foo", -1}}, &router).ok());
  // Out of range output port.
  EXPECT_FALSE(
      DynamicSourceRouter<string>::Create(2, {{"foo", 2}}, &router).ok());
  // Using reserved (last) output port.
  EXPECT_FALSE(
      DynamicSourceRouter<string>::Create(2, {{"foo", 1}}, &router).ok());
}

TEST(DynamicSourceRouterTest, ReconfigureToInvalidRouteMap) {
  std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
  TF_ASSERT_OK(DynamicSourceRouter<string>::Create(2, {{"foo", 0}}, &router));
  // Negative output port.
  EXPECT_FALSE(router->UpdateRoutes({{"foo", -1}}).ok());
  // Out of range output port.
  EXPECT_FALSE(router->UpdateRoutes({{"foo", 2}}).ok());
  // Using reserved (last) output port.
  EXPECT_FALSE(router->UpdateRoutes({{"foo", 1}}).ok());
}

TEST(DynamicSourceRouterTest, Basic) {
  std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
  DynamicSourceRouter<StoragePath>::Routes routes = {{"foo", 0}, {"bar", 1}};
  TF_ASSERT_OK(DynamicSourceRouter<string>::Create(4, routes, &router));
  EXPECT_EQ(routes, router->GetRoutes());
  std::vector<Source<StoragePath>*> output_ports = router->GetOutputPorts();
  ASSERT_EQ(4, output_ports.size());
  std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
  for (int i = 0; i < output_ports.size(); ++i) {
    std::unique_ptr<test_util::MockStoragePathTarget> target(
        new StrictMock<test_util::MockStoragePathTarget>);
    ConnectSourceToTarget(output_ports[i], target.get());
    targets.push_back(std::move(target));
  }

  // "foo" goes to port 0.
  EXPECT_CALL(*targets[0], SetAspiredVersions(
                               Eq("foo"), ElementsAre(ServableData<StoragePath>(
                                              {"foo", 7}, "data"))));
  router->SetAspiredVersions("foo",
                             {ServableData<StoragePath>({"foo", 7}, "data")});

  // "bar" goes to port 1.
  EXPECT_CALL(*targets[1], SetAspiredVersions(
                               Eq("bar"), ElementsAre(ServableData<StoragePath>(
                                              {"bar", 7}, "data"))));
  router->SetAspiredVersions("bar",
                             {ServableData<StoragePath>({"bar", 7}, "data")});

  // Servable whose name doesn't match any route goes to the last port (port 3).
  EXPECT_CALL(*targets[3],
              SetAspiredVersions(Eq("not_foo_or_bar"),
                                 ElementsAre(ServableData<StoragePath>(
                                     {"not_foo_or_bar", 7}, "data"))));
  router->SetAspiredVersions(
      "not_foo_or_bar",
      {ServableData<StoragePath>({"not_foo_or_bar", 7}, "data")});
}

TEST(DynamicSourceRouterTest, Reconfigure) {
  std::unique_ptr<DynamicSourceRouter<StoragePath>> router;
  TF_ASSERT_OK(DynamicSourceRouter<string>::Create(2, {{"foo", 0}}, &router));
  std::vector<Source<StoragePath>*> output_ports = router->GetOutputPorts();
  ASSERT_EQ(2, output_ports.size());
  std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
  for (int i = 0; i < output_ports.size(); ++i) {
    std::unique_ptr<test_util::MockStoragePathTarget> target(
        new StrictMock<test_util::MockStoragePathTarget>);
    ConnectSourceToTarget(output_ports[i], target.get());
    targets.push_back(std::move(target));
  }

  // Initially, "foo" goes to port 0 and "bar" goes to the default (port 1).
  EXPECT_CALL(*targets[0], SetAspiredVersions(
                               Eq("foo"), ElementsAre(ServableData<StoragePath>(
                                              {"foo", 7}, "data"))));
  router->SetAspiredVersions("foo",
                             {ServableData<StoragePath>({"foo", 7}, "data")});
  EXPECT_CALL(*targets[1], SetAspiredVersions(
                               Eq("bar"), ElementsAre(ServableData<StoragePath>(
                                              {"bar", 7}, "data"))));
  router->SetAspiredVersions("bar",
                             {ServableData<StoragePath>({"bar", 7}, "data")});

  TF_CHECK_OK(router->UpdateRoutes({{"bar", 0}}));

  // Now, the routes of "foo" and "bar" should be swapped.
  EXPECT_CALL(*targets[1], SetAspiredVersions(
                               Eq("foo"), ElementsAre(ServableData<StoragePath>(
                                              {"foo", 7}, "data"))));
  router->SetAspiredVersions("foo",
                             {ServableData<StoragePath>({"foo", 7}, "data")});
  EXPECT_CALL(*targets[0], SetAspiredVersions(
                               Eq("bar"), ElementsAre(ServableData<StoragePath>(
                                              {"bar", 7}, "data"))));
  router->SetAspiredVersions("bar",
                             {ServableData<StoragePath>({"bar", 7}, "data")});
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
