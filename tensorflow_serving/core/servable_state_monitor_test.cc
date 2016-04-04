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

#include "tensorflow_serving/core/servable_state_monitor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace tensorflow {
namespace serving {
namespace {

TEST(ServableStateMonitorTest, AddingStates) {
  auto bus = EventBus<ServableState>::CreateEventBus();
  ServableStateMonitor monitor(bus.get());
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_TRUE(monitor.GetVersionStates("foo").empty());
  EXPECT_TRUE(monitor.GetAllServableStates().empty());

  // Initial servable.
  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  bus->Publish(state_0);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(monitor.GetVersionStates("foo"), ElementsAre(Pair(42, state_0)));
  EXPECT_TRUE(monitor.GetVersionStates("bar").empty());
  EXPECT_THAT(
      monitor.GetAllServableStates(),
      UnorderedElementsAre(Pair("foo", ElementsAre(Pair(42, state_0)))));

  // New version of existing servable.
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  bus->Publish(state_1);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor.GetState(ServableId{"foo", 43}));
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(monitor.GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0), Pair(43, state_1)));
  EXPECT_TRUE(monitor.GetVersionStates("bar").empty());
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(Pair(
                  "foo", ElementsAre(Pair(42, state_0), Pair(43, state_1)))));

  // New servable name.
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 Status::OK()};
  bus->Publish(state_2);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor.GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor.GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor.GetState(ServableId{"bar", 7}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(monitor.GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0), Pair(43, state_1)));
  EXPECT_THAT(monitor.GetVersionStates("bar"), ElementsAre(Pair(7, state_2)));
  EXPECT_TRUE(monitor.GetVersionStates("baz").empty());
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(Pair("foo", ElementsAre(Pair(42, state_0),
                                                           Pair(43, state_1))),
                                   Pair("bar", ElementsAre(Pair(7, state_2)))));
}

TEST(ServableStateMonitorTest, UpdatingStates) {
  auto bus = EventBus<ServableState>::CreateEventBus();
  ServableStateMonitor monitor(bus.get());

  // Initial servables.
  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  bus->Publish(state_0);
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  bus->Publish(state_1);
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 Status::OK()};
  bus->Publish(state_2);
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(Pair("foo", ElementsAre(Pair(42, state_0),
                                                           Pair(43, state_1))),
                                   Pair("bar", ElementsAre(Pair(7, state_2)))));

  // Update one of them.
  const ServableState state_1_updated = {ServableId{"foo", 43},
                                         ServableState::ManagerState::kLoading,
                                         Status::OK()};
  bus->Publish(state_1_updated);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1_updated, *monitor.GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor.GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor.GetState(ServableId{"bar", 7}));
  EXPECT_THAT(monitor.GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0), Pair(43, state_1_updated)));
  EXPECT_THAT(monitor.GetVersionStates("bar"), ElementsAre(Pair(7, state_2)));
  EXPECT_THAT(
      monitor.GetAllServableStates(),
      UnorderedElementsAre(Pair("foo", ElementsAre(Pair(42, state_0),
                                                   Pair(43, state_1_updated))),
                           Pair("bar", ElementsAre(Pair(7, state_2)))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
