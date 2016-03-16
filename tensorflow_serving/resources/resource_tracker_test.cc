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

#include "tensorflow_serving/resources/resource_tracker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/test_util/test_util.h"

using ::tensorflow::serving::test_util::CreateProto;
using ::tensorflow::serving::test_util::EqualsProto;
using ::testing::Return;
using ::testing::NiceMock;

namespace tensorflow {
namespace serving {
namespace {

class ResourceTrackerTest : public ::testing::Test {
 protected:
  ResourceTrackerTest()
      : total_resources_(
            CreateProto<ResourceAllocation>("resource_quantities { "
                                            "  resource { "
                                            "    device: 'cpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 16 "
                                            "} "
                                            "resource_quantities { "
                                            "  resource { "
                                            "    device: 'gpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 16 "
                                            "} ")),
        tracker_(total_resources_,
                 std::unique_ptr<ResourceUtil>(
                     new ResourceUtil({{{"cpu", 1}, {"gpu", 2}}}))) {
    loader_0_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_0_, EstimateResources())
        .WillByDefault(Return(
            CreateProto<ResourceAllocation>("resource_quantities { "
                                            "  resource { "
                                            "    device: 'cpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 1 "
                                            "} "
                                            "resource_quantities { "
                                            "  resource { "
                                            "    device: 'gpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 3 "
                                            "} ")));

    loader_1_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_1_, EstimateResources())
        .WillByDefault(Return(
            CreateProto<ResourceAllocation>("resource_quantities { "
                                            "  resource { "
                                            "    device: 'cpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 5 "
                                            "} "
                                            "resource_quantities { "
                                            "  resource { "
                                            "    device: 'gpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 7 "
                                            "} ")));

    loader_2_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_2_, EstimateResources())
        .WillByDefault(Return(
            CreateProto<ResourceAllocation>("resource_quantities { "
                                            "  resource { "
                                            "    device: 'cpu' "
                                            "    device_instance { value: 0 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 15 "
                                            "} ")));

    // Disallow calls to Load()/Unload().
    for (auto* loader : {loader_0_.get(), loader_1_.get(), loader_2_.get()}) {
      EXPECT_CALL(*loader, Load()).Times(0);
      EXPECT_CALL(*loader, Unload()).Times(0);
    }
  }

  // The original total-resources valued provided to 'tracker_'.
  const ResourceAllocation total_resources_;

  // The object under testing.
  ResourceTracker tracker_;

  // Some mock loaders with specific resource estimates (see the constructor).
  std::unique_ptr<test_util::MockLoader> loader_0_;
  std::unique_ptr<test_util::MockLoader> loader_1_;
  std::unique_ptr<test_util::MockLoader> loader_2_;
};

TEST_F(ResourceTrackerTest, RecomputeUsedResources) {
  // Verify the initial state.
  EXPECT_THAT(tracker_.used_resources(), EqualsProto(""));
  EXPECT_THAT(tracker_.total_resources(), EqualsProto(total_resources_));

  // Recompute used resources for {loader_0_, loader_1_}.
  tracker_.RecomputeUsedResources({loader_0_.get(), loader_1_.get()});
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 6 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 10 "
                          "} "));
  EXPECT_THAT(tracker_.total_resources(), EqualsProto(total_resources_));

  // Recompute used resources for just {loader_0_}.
  tracker_.RecomputeUsedResources({loader_0_.get()});
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 1 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 3 "
                          "} "));
  EXPECT_THAT(tracker_.total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesSuccess) {
  tracker_.RecomputeUsedResources({loader_0_.get()});
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 1 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 3 "
                          "} "));

  // If just loader_0_ is loaded, loader_2_ should also fit.
  EXPECT_TRUE(tracker_.ReserveResources(*loader_2_));
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 16 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 3 "
                          "} "));
  EXPECT_THAT(tracker_.total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesFailure) {
  tracker_.RecomputeUsedResources({loader_1_.get()});
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 5 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 7 "
                          "} "));

  // If loader_1_ is loaded, there isn't room for loader_2_.
  EXPECT_FALSE(tracker_.ReserveResources(*loader_2_));
  // The used resources should remain unchanged (i.e. only reflect loader_1_).
  EXPECT_THAT(tracker_.used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 5 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 7 "
                          "} "));
  EXPECT_THAT(tracker_.total_resources(), EqualsProto(total_resources_));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
