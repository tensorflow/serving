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

#include "tensorflow_serving/core/availability_preserving_policy.h"

#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {
namespace {

// None unloadable (ready+unaspired); multiple loadable (new+aspired). Loads the
// highest loadable version.
TEST(AvailabilityPreservingPolicyTest, LoadsNewAspired) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 5}, LoaderHarness::State::kNew, true});

  AvailabilityPreservingPolicy policy;
  auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kLoad, action->action);
  EXPECT_EQ(5, action->id.version);
}

// Both unloadable and loadable versions present. Unloading doesn't compromise
// availability. Opts to unload.
TEST(AvailabilityPreservingPolicyTest, UnloadsNonAspiredFirst) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 4}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 5}, LoaderHarness::State::kNew, true});

  AvailabilityPreservingPolicy policy;
  auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(3, action->id.version);
}

// One unloadable. Nothing aspired so the goal is to unload all versions and
// lose availability. Unloads.
TEST(AvailabilityPreservingPolicyTest, UnloadsFirstNonAspiredWhenNoAspired) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kDisabled, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kError, false});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(2, action->id.version);
}

// None unloadable or loadable. Takes no action.
TEST(AvailabilityPreservingPolicyTest, ReturnsNoActionWhenNone) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 5}, LoaderHarness::State::kDisabled, false});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

// One unloadable; none loadable. Unloading would compromise availability. Takes
// no action.
TEST(AvailabilityPreservingPolicyTest, DoesNotUnloadWhenOtherNotReady) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kLoading, true});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

// One unloadable; none loadable. Unloading would compromise availability. Takes
// no action.
TEST(AvailabilityPreservingPolicyTest, DoesNotUnloadWhenOtherInError) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, true});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

// One unloadable; none loadable. Unloading doesn't compromise availability.
// Unloads.
TEST(AvailabilityPreservingPolicyTest, UnloadIfOtherReadyEvenIfLoading) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, true});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(1, action->id.version);
}

// One unloadable; none loadable. Unloading doesn't compromise availability.
// Unloads.
TEST(AvailabilityPreservingPolicyTest, UnloadIfOtherReadyEvenIfError) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, true});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(1, action->id.version);
}

// Multiple unloadable; none loadable. Availability is achieved despite no
// aspired versions being loaded, because one or more non-aspired versions are
// loaded. Unloading one non-aspired version doesn't compromise availability.
// Unloads the lowest such version.
TEST(AvailabilityPreservingPolicyTest, UnloadIfNoAspiredVersionsReady) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 2}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kError, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kLoading, true});

  AvailabilityPreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(1, action->id.version);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
