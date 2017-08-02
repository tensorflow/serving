/* Copyright 2017 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/core/aspired_version_policy.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow_serving/core/loader_harness.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

class AspiredVersionPolicyTest : public ::testing::Test {
 public:
  // Expose the protected AspiredVersionPolicy::GetHighestAspiredNewServableId()
  // method.
  optional<ServableId> GetHighestAspiredNewServableId(
      const std::vector<AspiredServableStateSnapshot>& all_versions) {
    return AspiredVersionPolicy::GetHighestAspiredNewServableId(all_versions);
  }
};

TEST_F(AspiredVersionPolicyTest, NoHighestAspiredNewServableId) {
  // No new aspired versions.
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 10}, LoaderHarness::State::kNew, false});
  versions.push_back({{"test", 9}, LoaderHarness::State::kUnloading, true});
  optional<ServableId> highest_aspired_new =
      GetHighestAspiredNewServableId(versions);
  ASSERT_FALSE(highest_aspired_new);
}

TEST_F(AspiredVersionPolicyTest, HighestAspiredNewServableId) {
  // Three new aspired versions and two other versions.
  std::vector<AspiredServableStateSnapshot> versions;
  versions.push_back({{"test", 10}, LoaderHarness::State::kNew, false});
  versions.push_back({{"test", 9}, LoaderHarness::State::kUnloading, true});
  versions.push_back({{"test", 1}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 5}, LoaderHarness::State::kNew, true});
  versions.push_back({{"test", 3}, LoaderHarness::State::kNew, true});

  optional<ServableId> highest_aspired_new =
      GetHighestAspiredNewServableId(versions);
  ASSERT_TRUE(highest_aspired_new);
  EXPECT_EQ("test", highest_aspired_new.value().name);
  EXPECT_EQ(5, highest_aspired_new.value().version);
}

}  // namespace serving
}  // namespace tensorflow
