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
#include "tensorflow_serving/test_util/fake_clock_env.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ServableStateAndTime = ServableStateMonitor::ServableStateAndTime;

TEST(ServableStateMonitorTest, AddingStates) {
  test_util::FakeClockEnv env(Env::Default());
  EventBus<ServableState>::Options bus_options;
  bus_options.env = &env;
  auto bus = EventBus<ServableState>::CreateEventBus(bus_options);

  ServableStateMonitor::Options monitor_options;
  monitor_options.max_count_log_events = 4;

  ServableStateMonitor monitor(bus.get(), monitor_options);
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_TRUE(monitor.GetVersionStates("foo").empty());
  EXPECT_TRUE(monitor.GetAllServableStates().empty());
  EXPECT_TRUE(monitor.GetBoundedLog().empty());

  // Initial servable.
  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  env.AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus->Publish(state_0);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(monitor.GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0_and_time)));
  EXPECT_TRUE(monitor.GetVersionStates("bar").empty());
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));
  EXPECT_THAT(monitor.GetBoundedLog(), ElementsAre(state_0_and_time));

  // New version of existing servable.
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  env.AdvanceByMicroseconds(2);
  const ServableStateAndTime state_1_and_time = {state_1, 3};
  bus->Publish(state_1);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor.GetState(ServableId{"foo", 43}));
  EXPECT_FALSE(monitor.GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(
      monitor.GetVersionStates("foo"),
      ElementsAre(Pair(42, state_0_and_time), Pair(43, state_1_and_time)));
  EXPECT_TRUE(monitor.GetVersionStates("bar").empty());
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time),
                                          Pair(43, state_1_and_time)))));
  EXPECT_THAT(monitor.GetBoundedLog(),
              ElementsAre(state_0_and_time, state_1_and_time));

  // New servable name.
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 Status::OK()};
  env.AdvanceByMicroseconds(4);
  const ServableStateAndTime state_2_and_time = {state_2, 7};
  bus->Publish(state_2);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor.GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor.GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor.GetState(ServableId{"bar", 7}));
  EXPECT_FALSE(monitor.GetState(ServableId{"bar", 42}));
  EXPECT_THAT(
      monitor.GetVersionStates("foo"),
      ElementsAre(Pair(42, state_0_and_time), Pair(43, state_1_and_time)));
  EXPECT_THAT(monitor.GetVersionStates("bar"),
              ElementsAre(Pair(7, state_2_and_time)));
  EXPECT_TRUE(monitor.GetVersionStates("baz").empty());
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time),
                                          Pair(43, state_1_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));

  EXPECT_THAT(
      monitor.GetBoundedLog(),
      ElementsAre(state_0_and_time, state_1_and_time, state_2_and_time));
}

TEST(ServableStateMonitorTest, UpdatingStates) {
  test_util::FakeClockEnv env(Env::Default());
  EventBus<ServableState>::Options bus_options;
  bus_options.env = &env;
  auto bus = EventBus<ServableState>::CreateEventBus(bus_options);

  ServableStateMonitor::Options monitor_options;
  monitor_options.max_count_log_events = 3;
  ServableStateMonitor monitor(bus.get(), monitor_options);

  // Initial servables.
  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  env.AdvanceByMicroseconds(4);
  const ServableStateAndTime state_0_and_time = {state_0, 4};
  bus->Publish(state_0);
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  const ServableStateAndTime state_1_and_time = {state_1, 4};
  bus->Publish(state_1);
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 Status::OK()};
  const ServableStateAndTime state_2_and_time = {state_2, 4};
  bus->Publish(state_2);
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time),
                                          Pair(43, state_1_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));
  EXPECT_THAT(
      monitor.GetBoundedLog(),
      ElementsAre(state_0_and_time, state_1_and_time, state_2_and_time));

  // Update one of them.
  const ServableState state_1_updated = {ServableId{"foo", 43},
                                         ServableState::ManagerState::kLoading,
                                         Status::OK()};
  env.AdvanceByMicroseconds(4);
  const ServableStateAndTime state_1_updated_and_time = {state_1_updated, 8};
  bus->Publish(state_1_updated);
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor.GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor.GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1_updated, *monitor.GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor.GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor.GetState(ServableId{"bar", 7}));
  EXPECT_THAT(monitor.GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0_and_time),
                          Pair(43, state_1_updated_and_time)));
  EXPECT_THAT(monitor.GetVersionStates("bar"),
              ElementsAre(Pair(7, state_2_and_time)));
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time),
                                          Pair(43, state_1_updated_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));

  // The max count for events logged in the bounded log is 3, so the first entry
  // corresponding to state_0 is removed and an entry is added for
  // state_1_updated.
  EXPECT_THAT(monitor.GetBoundedLog(),
              ElementsAre(state_1_and_time, state_2_and_time,
                          state_1_updated_and_time));
}

TEST(ServableStateMonitorTest, DisableBoundedLogging) {
  test_util::FakeClockEnv env(Env::Default());
  EventBus<ServableState>::Options bus_options;
  bus_options.env = &env;
  auto bus = EventBus<ServableState>::CreateEventBus(bus_options);

  // The default value for max_count_log_events in options is 0, which disables
  // logging.
  ServableStateMonitor monitor(bus.get());
  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  env.AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus->Publish(state_0);
  EXPECT_THAT(monitor.GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));
  EXPECT_TRUE(monitor.GetBoundedLog().empty());
}

TEST(ServableStateMonitorTest, GetLiveServableStates) {
  test_util::FakeClockEnv env(Env::Default());
  EventBus<ServableState>::Options bus_options;
  bus_options.env = &env;
  auto bus = EventBus<ServableState>::CreateEventBus(bus_options);
  ServableStateMonitor monitor(bus.get());

  const ServableState state_0 = {
      ServableId{"foo", 42}, ServableState::ManagerState::kStart, Status::OK()};
  env.AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus->Publish(state_0);
  EXPECT_THAT(monitor.GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));

  const ServableState state_1 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kAvailable,
                                 Status::OK()};
  env.AdvanceByMicroseconds(1);
  const ServableStateAndTime state_1_and_time = {state_1, 2};
  bus->Publish(state_1);
  EXPECT_THAT(monitor.GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_1_and_time)))));

  // Servable {foo, 42} moves to state kEnd and is removed from the live states
  // servables.
  const ServableState state_0_update = {
      ServableId{"foo", 42}, ServableState::ManagerState::kEnd, Status::OK()};
  env.AdvanceByMicroseconds(1);
  bus->Publish(state_0_update);
  EXPECT_THAT(monitor.GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("bar", ElementsAre(Pair(7, state_1_and_time)))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
