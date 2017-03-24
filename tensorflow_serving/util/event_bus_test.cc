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

#include "tensorflow_serving/util/event_bus.h"

#include <gtest/gtest.h>
#include "tensorflow/contrib/batching/test_util/fake_clock_env.h"

namespace tensorflow {
namespace serving {
namespace {

typedef EventBus<int> IntEventBus;

TEST(EventBusTest, PublishNoSubscribers) {
  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus();
  bus->Publish(42);
}

// Tests a typical full lifecycle
TEST(EventBusTest, FullLifecycleTest) {
  test_util::FakeClockEnv env(Env::Default());
  IntEventBus::Options bus_options;
  bus_options.env = &env;

  // Set up a bus with a single subscriber.
  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus(bus_options);
  int value = 1;
  uint64 value_timestamp = 0;
  IntEventBus::Callback callback =
      [&value, &value_timestamp](IntEventBus::EventAndTime event_and_time) {
        value += event_and_time.event;
        value_timestamp = event_and_time.event_time_micros;
      };
  std::unique_ptr<IntEventBus::Subscription> subscription =
      bus->Subscribe(callback);

  // Publish once. Confirm the subscriber was called with the event and the
  // corresponding timestamp for the published event was set.
  env.AdvanceByMicroseconds(1);
  bus->Publish(2);
  ASSERT_EQ(3, value);
  ASSERT_EQ(1, value_timestamp);

  // Set up a second subscriber
  int other_value = 100;
  int other_value_timestamp = 0;
  IntEventBus::Callback second_callback =
      [&other_value,
       &other_value_timestamp](IntEventBus::EventAndTime event_and_time) {
        other_value += event_and_time.event;
        other_value_timestamp = event_and_time.event_time_micros;
      };
  std::unique_ptr<IntEventBus::Subscription> other_subscription =
      bus->Subscribe(second_callback);

  // Publish a second time. Confirm that both subscribers were called and that
  // corresponding timestamps for the published events were set.
  env.AdvanceByMicroseconds(2);
  bus->Publish(10);
  EXPECT_EQ(13, value);
  EXPECT_EQ(3, value_timestamp);
  EXPECT_EQ(110, other_value);
  EXPECT_EQ(3, other_value_timestamp);

  subscription.reset();

  // Publish again and confirm that only the second subscriber was called.
  env.AdvanceByMicroseconds(3);
  bus->Publish(20);
  EXPECT_EQ(13, value);
  EXPECT_EQ(3, value_timestamp);
  EXPECT_EQ(130, other_value);
  EXPECT_EQ(6, other_value_timestamp);

  // Explicitly test that the EventBus can be destroyed before the last
  // subscriber.
  bus.reset();
  other_subscription.reset();
}

// Tests automatic unsubscribing behavior with the RAII pattern.
TEST(EventBusTest, TestAutomaticUnsubscribing) {
  test_util::FakeClockEnv env(Env::Default());
  IntEventBus::Options bus_options;
  bus_options.env = &env;

  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus(bus_options);
  int value = 1;
  int value_timestamp = 0;
  IntEventBus::Callback callback =
      [&value, &value_timestamp](IntEventBus::EventAndTime event_and_time) {
        value += event_and_time.event;
        value_timestamp += event_and_time.event_time_micros;
      };
  {
    std::unique_ptr<IntEventBus::Subscription> subscription =
        bus->Subscribe(callback);

    // Publish once. Confirm the subscriber was called with the event and the
    // corresponding timestamp for the published event was set.
    env.AdvanceByMicroseconds(3);
    bus->Publish(2);
    EXPECT_EQ(3, value);
    EXPECT_EQ(3, value_timestamp);
  }

  // Publish again after the Subscription is no longer in scope and confirm that
  // the subscriber was not called.
  env.AdvanceByMicroseconds(1);
  bus->Publish(2);
  EXPECT_EQ(3, value);
  EXPECT_EQ(3, value_timestamp);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
