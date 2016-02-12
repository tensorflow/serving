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

#include "tensorflow_serving/core/eager_load_policy.h"

#include <gtest/gtest.h>
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {
namespace {

// Test that the first new and aspired version is loaded first.
TEST(EagerLoadPolicy, LoadsFirstAspired) {
  std::vector<ServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 2}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 4}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 5}, LoaderHarness::State::kReady, false});

  EagerLoadPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(VersionPolicy::Action::kLoad, action->action);
  EXPECT_EQ(4, action->id.version);
}

// Test that the first non-aspired version is unloaded when there are none to
// load.
TEST(EagerLoadPolicy, UnLoadsFirstNonAspiredWhenNoneToLoad) {
  std::vector<ServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 2}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kReady, false});
  versions.push_back({{"test", 4}, LoaderHarness::State::kDisabled, false});
  versions.push_back({{"test", 5}, LoaderHarness::State::kReady, false});

  EagerLoadPolicy policy;
  const auto action = policy.GetNextAction(versions);
  ASSERT_TRUE(action);
  EXPECT_EQ(VersionPolicy::Action::kUnload, action->action);
  EXPECT_EQ(3, action->id.version);
}

// Test that no action is returned (empty optional) when there are no versions
// needing loading or unloading.
TEST(EagerLoadPolicy, ReturnsNoActionWhenNone) {
  std::vector<ServableStateSnapshot> versions;
  versions.push_back({{"test", 1}, LoaderHarness::State::kReady, true});
  versions.push_back({{"test", 2}, LoaderHarness::State::kError, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kLoading, true});
  versions.push_back({{"test", 4}, LoaderHarness::State::kUnloading, false});
  versions.push_back({{"test", 5}, LoaderHarness::State::kDisabled, false});

  EagerLoadPolicy policy;
  const auto action = policy.GetNextAction(versions);
  EXPECT_FALSE(action);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
