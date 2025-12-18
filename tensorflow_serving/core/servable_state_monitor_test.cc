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

#include <map>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/util/event_bus.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ServableStateAndTime = ServableStateMonitor::ServableStateAndTime;

class ServableStateMonitorTest : public ::testing::Test {
 protected:
  ServableStateMonitorTest() {
    env_ = std::make_unique<test_util::FakeClockEnv>(Env::Default());
    EventBus<ServableState>::Options bus_options;
    bus_options.env = env_.get();
    bus_ = EventBus<ServableState>::CreateEventBus(bus_options);
  }
  void CreateMonitor(int max_count_log_events = 0) {
    ServableStateMonitor::Options monitor_options;
    monitor_options.max_count_log_events = max_count_log_events;
    monitor_ =
        std::make_unique<ServableStateMonitor>(bus_.get(), monitor_options);
  }
  std::unique_ptr<test_util::FakeClockEnv> env_;
  std::shared_ptr<EventBus<ServableState>> bus_;
  std::unique_ptr<ServableStateMonitor> monitor_;
};

TEST_F(ServableStateMonitorTest, AddingStates) {
  CreateMonitor(/*max_count_log_events=*/4);
  ServableState notified_state;
  monitor_->Notify([&](const ServableState& servable_state) {
    notified_state = servable_state;
  });
  EXPECT_FALSE(monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_TRUE(monitor_->GetVersionStates("foo").empty());
  EXPECT_TRUE(monitor_->GetAllServableStates().empty());
  EXPECT_TRUE(monitor_->GetBoundedLog().empty());

  // Initial servable.
  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, notified_state);
  EXPECT_FALSE(monitor_->GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor_->GetState(ServableId{"bar", 42}));
  EXPECT_THAT(monitor_->GetVersionStates("foo"),
              ElementsAre(Pair(42, state_0_and_time)));
  EXPECT_TRUE(monitor_->GetVersionStates("bar").empty());
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));
  EXPECT_THAT(monitor_->GetBoundedLog(), ElementsAre(state_0_and_time));

  // New version of existing servable.
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  env_->AdvanceByMicroseconds(2);
  const ServableStateAndTime state_1_and_time = {state_1, 3};
  bus_->Publish(state_1);
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor_->GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor_->GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, notified_state);
  EXPECT_FALSE(monitor_->GetState(ServableId{"foo", 99}));
  EXPECT_FALSE(monitor_->GetState(ServableId{"bar", 42}));
  EXPECT_THAT(
      monitor_->GetVersionStates("foo"),
      ElementsAre(Pair(43, state_1_and_time), Pair(42, state_0_and_time)));
  EXPECT_TRUE(monitor_->GetVersionStates("bar").empty());
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(43, state_1_and_time),
                                          Pair(42, state_0_and_time)))));
  EXPECT_THAT(monitor_->GetBoundedLog(),
              ElementsAre(state_0_and_time, state_1_and_time));

  // New servable name.
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(4);
  const ServableStateAndTime state_2_and_time = {state_2, 7};
  bus_->Publish(state_2);
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor_->GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1, *monitor_->GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor_->GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor_->GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, notified_state);
  EXPECT_FALSE(monitor_->GetState(ServableId{"bar", 42}));
  EXPECT_THAT(
      monitor_->GetVersionStates("foo"),
      ElementsAre(Pair(43, state_1_and_time), Pair(42, state_0_and_time)));
  EXPECT_THAT(monitor_->GetVersionStates("bar"),
              ElementsAre(Pair(7, state_2_and_time)));
  EXPECT_TRUE(monitor_->GetVersionStates("baz").empty());
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(43, state_1_and_time),
                                          Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));

  EXPECT_THAT(
      monitor_->GetBoundedLog(),
      ElementsAre(state_0_and_time, state_1_and_time, state_2_and_time));
}

TEST_F(ServableStateMonitorTest, UpdatingStates) {
  CreateMonitor(/*max_count_log_events=*/3);

  // Initial servables.
  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(4);
  const ServableStateAndTime state_0_and_time = {state_0, 4};
  bus_->Publish(state_0);
  const ServableState state_1 = {ServableId{"foo", 43},
                                 ServableState::ManagerState::kAvailable,
                                 errors::Unknown("error")};
  const ServableStateAndTime state_1_and_time = {state_1, 4};
  bus_->Publish(state_1);
  const ServableState state_2 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kUnloading,
                                 absl::OkStatus()};
  const ServableStateAndTime state_2_and_time = {state_2, 4};
  bus_->Publish(state_2);
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(43, state_1_and_time),
                                          Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));
  EXPECT_THAT(
      monitor_->GetBoundedLog(),
      ElementsAre(state_0_and_time, state_1_and_time, state_2_and_time));

  // Update one of them.
  const ServableState state_1_updated = {ServableId{"foo", 43},
                                         ServableState::ManagerState::kLoading,
                                         absl::OkStatus()};
  env_->AdvanceByMicroseconds(4);
  const ServableStateAndTime state_1_updated_and_time = {state_1_updated, 8};
  bus_->Publish(state_1_updated);
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 42}));
  EXPECT_EQ(state_0, *monitor_->GetState(ServableId{"foo", 42}));
  ASSERT_TRUE(monitor_->GetState(ServableId{"foo", 43}));
  EXPECT_EQ(state_1_updated, *monitor_->GetState(ServableId{"foo", 43}));
  ASSERT_TRUE(monitor_->GetState(ServableId{"bar", 7}));
  EXPECT_EQ(state_2, *monitor_->GetState(ServableId{"bar", 7}));
  EXPECT_THAT(monitor_->GetVersionStates("foo"),
              ElementsAre(Pair(43, state_1_updated_and_time),
                          Pair(42, state_0_and_time)));
  EXPECT_THAT(monitor_->GetVersionStates("bar"),
              ElementsAre(Pair(7, state_2_and_time)));
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(43, state_1_updated_and_time),
                                          Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_2_and_time)))));

  // The max count for events logged in the bounded log is 3, so the first entry
  // corresponding to state_0 is removed and an entry is added for
  // state_1_updated.
  EXPECT_THAT(monitor_->GetBoundedLog(),
              ElementsAre(state_1_and_time, state_2_and_time,
                          state_1_updated_and_time));
}

TEST_F(ServableStateMonitorTest, DisableBoundedLogging) {
  // The default value for max_count_log_events in options is 0, which disables
  // logging.
  CreateMonitor();
  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));
  EXPECT_TRUE(monitor_->GetBoundedLog().empty());
}

TEST_F(ServableStateMonitorTest, GetLiveServableStates) {
  CreateMonitor();

  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));

  const ServableState state_1 = {ServableId{"bar", 7},
                                 ServableState::ManagerState::kAvailable,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_1_and_time = {state_1, 2};
  bus_->Publish(state_1);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(7, state_1_and_time)))));

  // Servable {foo, 42} moves to state kEnd and is removed from the live states
  // servables.
  const ServableState state_0_update = {ServableId{"foo", 42},
                                        ServableState::ManagerState::kEnd,
                                        absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  bus_->Publish(state_0_update);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("bar", ElementsAre(Pair(7, state_1_and_time)))));
}

TEST_F(ServableStateMonitorTest, GetAvailableServableStates) {
  CreateMonitor();

  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  EXPECT_THAT(monitor_->GetAvailableServableStates(), testing::IsEmpty());

  env_->AdvanceByMicroseconds(1);
  std::vector<ServableStateAndTime> servable_state_and_time;
  for (const auto& servable_id : {ServableId{"bar", 6}, ServableId{"bar", 7}}) {
    const ServableState state = {
        servable_id, ServableState::ManagerState::kAvailable, absl::OkStatus()};
    const ServableStateAndTime state_and_time = {state, 2};
    servable_state_and_time.push_back({state, 2});
    bus_->Publish(state);
  }

  EXPECT_THAT(monitor_->GetAvailableServableStates(),
              UnorderedElementsAre("bar"));

  // Servable {bar, 6} moves to state kUnloading and is removed from available
  // servable states.
  const ServableState state_0_update = {ServableId{"bar", 6},
                                        ServableState::ManagerState::kUnloading,
                                        absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  bus_->Publish(state_0_update);
  EXPECT_THAT(monitor_->GetAvailableServableStates(),
              UnorderedElementsAre("bar"));
  // Servable {bar, 7} moves to state kEnd and is removed from available
  // servable states.
  const ServableState state_1_update = {ServableId{"bar", 7},
                                        ServableState::ManagerState::kEnd,
                                        absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  bus_->Publish(state_1_update);
  // No available state now.
  EXPECT_THAT(monitor_->GetAvailableServableStates(), ::testing::IsEmpty());
}

TEST_F(ServableStateMonitorTest, VersionMapDescendingOrder) {
  CreateMonitor();

  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kStart,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));

  const ServableState state_1 = {ServableId{"foo", 7},
                                 ServableState::ManagerState::kAvailable,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_1_and_time = {state_1, 2};
  bus_->Publish(state_1);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              ElementsAre(Pair("foo", ElementsAre(Pair(42, state_0_and_time),
                                                  Pair(7, state_1_and_time)))));
}

TEST_F(ServableStateMonitorTest, ForgetUnloadedServableStates) {
  CreateMonitor();

  const ServableState state_0 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kAvailable,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_0_and_time = {state_0, 1};
  bus_->Publish(state_0);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time)))));

  const ServableState state_1 = {ServableId{"bar", 1},
                                 ServableState::ManagerState::kAvailable,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_1_and_time = {state_1, 2};
  bus_->Publish(state_1);
  EXPECT_THAT(monitor_->GetLiveServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_0_and_time))),
                  Pair("bar", ElementsAre(Pair(1, state_1_and_time)))));

  const ServableState state_2 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kUnloading,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_2_and_time = {state_2, 3};
  bus_->Publish(state_2);
  monitor_->ForgetUnloadedServableStates();
  // "foo" state should still be recorded since it hasn't reached kEnd.
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_2_and_time))),
                  Pair("bar", ElementsAre(Pair(1, state_1_and_time)))));

  const ServableState state_3 = {ServableId{"foo", 42},
                                 ServableState::ManagerState::kEnd,
                                 absl::OkStatus()};
  env_->AdvanceByMicroseconds(1);
  const ServableStateAndTime state_3_and_time = {state_3, 4};
  bus_->Publish(state_3);
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", ElementsAre(Pair(42, state_3_and_time))),
                  Pair("bar", ElementsAre(Pair(1, state_1_and_time)))));
  monitor_->ForgetUnloadedServableStates();
  EXPECT_THAT(monitor_->GetAllServableStates(),
              UnorderedElementsAre(
                  Pair("foo", IsEmpty()),
                  Pair("bar", ElementsAre(Pair(1, state_1_and_time)))));
}

TEST_F(ServableStateMonitorTest, NotifyWhenServablesReachStateZeroServables) {
  CreateMonitor();
  const std::vector<ServableRequest> servables = {};

  using ManagerState = ServableState::ManagerState;

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_TRUE(reached);
        EXPECT_THAT(states_reached, IsEmpty());
        notified.Notify();
      });
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest,
       NotifyWhenServablesReachStateSpecificAvailable) {
  CreateMonitor();
  std::vector<ServableRequest> servables;
  const ServableId specific_goal_state_id = {"specific_goal_state", 42};
  servables.push_back(ServableRequest::FromId(specific_goal_state_id));

  using ManagerState = ServableState::ManagerState;
  const ServableState specific_goal_state = {
      specific_goal_state_id, ManagerState::kAvailable, absl::OkStatus()};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_TRUE(reached);
        EXPECT_THAT(states_reached, UnorderedElementsAre(Pair(
                                        ServableId{"specific_goal_state", 42},
                                        ManagerState::kAvailable)));
        notified.Notify();
      });
  bus_->Publish(specific_goal_state);
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest, NotifyWhenServablesReachStateSpecificError) {
  CreateMonitor();
  std::vector<ServableRequest> servables;
  const ServableId specific_error_state_id = {"specific_error_state", 42};
  servables.push_back(ServableRequest::FromId(specific_error_state_id));

  using ManagerState = ServableState::ManagerState;
  const ServableState specific_error_state = {
      specific_error_state_id, ManagerState::kEnd, errors::Internal("error")};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_FALSE(reached);
        EXPECT_THAT(states_reached,
                    UnorderedElementsAre(
                        Pair(specific_error_state_id, ManagerState::kEnd)));
        notified.Notify();
      });
  bus_->Publish(specific_error_state);
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest,
       NotifyWhenServablesReachStateServableLatestAvailable) {
  CreateMonitor();
  std::vector<ServableRequest> servables;
  servables.push_back(ServableRequest::Latest("servable_stream"));
  const ServableId servable_stream_available_state_id = {"servable_stream", 42};

  using ManagerState = ServableState::ManagerState;
  const ServableState servable_stream_available_state = {
      servable_stream_available_state_id, ManagerState::kAvailable,
      absl::OkStatus()};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_TRUE(reached);
        EXPECT_THAT(states_reached, UnorderedElementsAre(
                                        Pair(servable_stream_available_state_id,
                                             ManagerState::kAvailable)));
        notified.Notify();
      });
  bus_->Publish(servable_stream_available_state);
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest, NotifyWhenServablesReachStateLatestError) {
  CreateMonitor();
  std::vector<ServableRequest> servables;
  servables.push_back(ServableRequest::Latest("servable_stream"));
  const ServableId servable_stream_error_state_id = {"servable_stream", 7};

  using ManagerState = ServableState::ManagerState;
  const ServableState servable_stream_error_state = {
      servable_stream_error_state_id, ManagerState::kEnd,
      errors::Internal("error")};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_FALSE(reached);
        EXPECT_THAT(states_reached,
                    UnorderedElementsAre(Pair(servable_stream_error_state_id,
                                              ManagerState::kEnd)));
        notified.Notify();
      });
  bus_->Publish(servable_stream_error_state);
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest,
       NotifyWhenServablesReachStateFullFunctionality) {
  using ManagerState = ServableState::ManagerState;

  CreateMonitor();
  std::vector<ServableRequest> servables;
  const ServableId specific_goal_state_id = {"specific_goal_state", 42};
  servables.push_back(ServableRequest::FromId(specific_goal_state_id));
  const ServableId specific_error_state_id = {"specific_error_state", 42};
  servables.push_back(ServableRequest::FromId(specific_error_state_id));
  servables.push_back(ServableRequest::Latest("servable_stream"));
  const ServableId servable_stream_id = {"servable_stream", 7};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        EXPECT_FALSE(reached);
        EXPECT_THAT(states_reached,
                    UnorderedElementsAre(
                        Pair(specific_goal_state_id, ManagerState::kAvailable),
                        Pair(specific_error_state_id, ManagerState::kEnd),
                        Pair(servable_stream_id, ManagerState::kAvailable)));
        notified.Notify();
      });

  const ServableState specific_goal_state = {
      specific_goal_state_id, ManagerState::kAvailable, absl::OkStatus()};
  const ServableState specific_error_state = {
      specific_error_state_id, ManagerState::kEnd, errors::Internal("error")};
  const ServableState servable_stream_state = {
      servable_stream_id, ManagerState::kAvailable, absl::OkStatus()};

  bus_->Publish(specific_goal_state);
  ASSERT_FALSE(notified.HasBeenNotified());
  bus_->Publish(specific_error_state);
  ASSERT_FALSE(notified.HasBeenNotified());
  bus_->Publish(servable_stream_state);
  notified.WaitForNotification();
}

TEST_F(ServableStateMonitorTest,
       NotifyWhenServablesReachStateOnlyNotifiedOnce) {
  CreateMonitor();
  std::vector<ServableRequest> servables;
  const ServableId specific_goal_state_id = {"specific_goal_state", 42};
  servables.push_back(ServableRequest::FromId(specific_goal_state_id));

  using ManagerState = ServableState::ManagerState;
  const ServableState specific_goal_state = {
      specific_goal_state_id, ManagerState::kAvailable, absl::OkStatus()};

  absl::Notification notified;
  monitor_->NotifyWhenServablesReachState(
      servables, ManagerState::kAvailable,
      [&](const bool reached,
          std::map<ServableId, ManagerState> states_reached) {
        // Will fail if this function is called twice.
        ASSERT_FALSE(notified.HasBeenNotified());
        EXPECT_TRUE(reached);
        EXPECT_THAT(states_reached, UnorderedElementsAre(Pair(
                                        ServableId{"specific_goal_state", 42},
                                        ManagerState::kAvailable)));
        notified.Notify();
      });
  bus_->Publish(specific_goal_state);
  notified.WaitForNotification();
  bus_->Publish(specific_goal_state);
}

TEST_F(ServableStateMonitorTest,
       WaitUntilServablesReachStateFullFunctionality) {
  using ManagerState = ServableState::ManagerState;

  CreateMonitor();
  std::vector<ServableRequest> servables;
  const ServableId specific_goal_state_id = {"specific_goal_state", 42};
  servables.push_back(ServableRequest::FromId(specific_goal_state_id));
  const ServableId specific_error_state_id = {"specific_error_state", 42};
  servables.push_back(ServableRequest::FromId(specific_error_state_id));
  servables.push_back(ServableRequest::Latest("servable_stream"));
  const ServableId servable_stream_id = {"servable_stream", 7};

  const ServableState specific_goal_state = {
      specific_goal_state_id, ManagerState::kAvailable, absl::OkStatus()};
  const ServableState specific_error_state = {
      specific_error_state_id, ManagerState::kEnd, errors::Internal("error")};
  const ServableState servable_stream_state = {
      servable_stream_id, ManagerState::kAvailable, absl::OkStatus()};

  bus_->Publish(specific_goal_state);
  bus_->Publish(specific_error_state);

  std::map<ServableId, ManagerState> states_reached;
  absl::Notification waiting_done;
  std::unique_ptr<Thread> wait_till_servable_state_reached(
      Env::Default()->StartThread({}, "WaitUntilServablesReachState", [&]() {
        EXPECT_FALSE(monitor_->WaitUntilServablesReachState(
            servables, ManagerState::kAvailable, &states_reached));
        EXPECT_THAT(states_reached,
                    UnorderedElementsAre(
                        Pair(specific_goal_state_id, ManagerState::kAvailable),
                        Pair(specific_error_state_id, ManagerState::kEnd),
                        Pair(servable_stream_id, ManagerState::kAvailable)));
        waiting_done.Notify();
      }));
  // We publish till waiting is finished, otherwise we could publish before we
  // could start waiting.
  while (!waiting_done.HasBeenNotified()) {
    bus_->Publish(servable_stream_state);
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
