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

#include "tensorflow_serving/core/resource_preserving_policy.h"

#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {
namespace {

// Test that the first ready and non-aspired version is unloaded first.
TEST(ResourcePreservingPolicyTest, UnloadsFirstNonAspired) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 4}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 5}, LoaderHarness::State::kReady, false});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(3, action->id.version);
}

// Test that the aspired new version with the highest version is loaded when
// there are none to unload.
TEST(ResourcePreservingPolicyTest, LoadsFirstAspiredWhenNoneToUnload) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 2}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kDisabled, false});
  versions.push_back({{"test", 5}, LoaderHarness::State::kNew, true});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kLoad, action->action);
  EXPECT_EQ(5, action->id.version);
}

// Test that no action is returned (empty optional) when there are no versions
// needing loading or unloading.
TEST(ResourcePreservingPolicyTest, ReturnsNoActionWhenNone) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 5}, LoaderHarness::State::kDisabled, false});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

TEST(ResourcePreservingPolicyTest, DoesNotLoadWhenOthersStillUnloading) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back(
      {{"test", 1}, LoaderHarness::State::kUnloadRequested, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kNew, true});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

TEST(ResourcePreservingPolicyTest, LoadIfUnaspiredIsError) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kError, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kNew, true});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kLoad, action->action);
  EXPECT_EQ(2, action->id.version);
}

TEST(ResourcePreservingPolicyTest, ErrorAndUnloadRequestedPreventLoading) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kError, false});
  versions.push_back(
      {{"test", 2}, LoaderHarness::State::kUnloadRequested, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kNew, true});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

TEST(ResourcePreservingPolicyTest, ErrorAndDisabledAllowLoading) {
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kError, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kDisabled, false});
  versions.push_back({{"test", 3}, LoaderHarness::State::kNew, true});

  ResourcePreservingPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(AspiredVersionPolicy::Action::kLoad, action->action);
  EXPECT_EQ(3, action->id.version);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
