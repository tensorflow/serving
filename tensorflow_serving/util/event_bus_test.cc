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

namespace tensorflow {
namespace serving {

typedef EventBus<int> IntEventBus;

TEST(EventBusTest, PublishNoSubscribers) {
  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus();
  bus->Publish(42);
}

// Tests a typical full lifecycle
TEST(EventBusTest, FullLifecycleTest) {
  // Set up a bus with a single subscriber.
  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus();
  int value = 1;
  IntEventBus::Callback callback = [&value](int event) { value += event; };
  std::unique_ptr<IntEventBus::Subscription> subscription =
      bus->Subscribe(callback);

  // Publish once and confirm the subscriber was called with the event.
  bus->Publish(2);
  ASSERT_EQ(3, value);

  // Set up a second subscriber
  int other_value = 100;
  IntEventBus::Callback second_callback = [&other_value](int event) {
    other_value += event;
  };
  std::unique_ptr<IntEventBus::Subscription> other_subscription =
      bus->Subscribe(second_callback);

  // Publish a second time and confirm that both subscribers were called.
  bus->Publish(10);
  EXPECT_EQ(13, value);
  EXPECT_EQ(110, other_value);

  subscription.reset();

  // Publish again and confirm that only the second subscriber was called.
  bus->Publish(20);
  EXPECT_EQ(13, value);
  EXPECT_EQ(130, other_value);

  // Explicitly test that the EventBus can be destroyed before the last
  // subscriber.
  bus.reset();
  other_subscription.reset();
}

// Tests automatic unsubscribing behavior with the RAII pattern.
TEST(EventBusTest, TestAutomaticUnsubscribing) {
  std::shared_ptr<IntEventBus> bus = IntEventBus::CreateEventBus();
  int value = 1;
  IntEventBus::Callback callback = [&value](int event) { value += event; };
  {
    std::unique_ptr<IntEventBus::Subscription> subscription =
        bus->Subscribe(callback);

    // Publish once and confirm the subscriber was called with the event.
    bus->Publish(2);
    EXPECT_EQ(3, value);
  }

  // Publish again after the Subscription is no longer in scope
  // and confirm that the subscriber was not called.
  bus->Publish(2);
  EXPECT_EQ(3, value);
}

}  // namespace serving
}  // namespace tensorflow
