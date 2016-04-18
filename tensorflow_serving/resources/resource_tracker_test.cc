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

#include <algorithm>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/any_ptr.h"

using ::tensorflow::serving::test_util::CreateProto;
using ::tensorflow::serving::test_util::EqualsProto;
using ::testing::_;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;

namespace tensorflow {
namespace serving {
namespace {

class ResourceTrackerTest : public ::testing::Test {
 protected:
  ResourceTrackerTest()
      : total_resources_(
            CreateProto<ResourceAllocation>("resource_quantities { "
                                            "  resource { "
                                            "    device: 'main' "
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
                                            "} "
                                            "resource_quantities { "
                                            "  resource { "
                                            "    device: 'gpu' "
                                            "    device_instance { value: 1 } "
                                            "    kind: 'ram' "
                                            "  } "
                                            "  quantity: 16 "
                                            "} ")) {
    std::unique_ptr<ResourceUtil> util(
        new ResourceUtil({{{"main", 1}, {"gpu", 2}}}));
    TF_CHECK_OK(ResourceTracker::Create(
        total_resources_, std::unique_ptr<ResourceUtil>(std::move(util)),
        &tracker_));

    loader_0_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_0_, EstimateResources(_))
        .WillByDefault(Invoke([](ResourceAllocation* estimate) {
          *estimate = CreateProto<ResourceAllocation>(
              "resource_quantities { "
              "  resource { "
              "    device: 'main' "
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
              "} ");
          return Status::OK();
        }));

    loader_1_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_1_, EstimateResources(_))
        .WillByDefault(Invoke([](ResourceAllocation* estimate) {
          *estimate = CreateProto<ResourceAllocation>(
              "resource_quantities { "
              "  resource { "
              "    device: 'main' "
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
              "} ");
          return Status::OK();
        }));

    loader_2_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_2_, EstimateResources(_))
        .WillByDefault(Invoke([](ResourceAllocation* estimate) {
          *estimate = CreateProto<ResourceAllocation>(
              "resource_quantities { "
              "  resource { "
              "    device: 'main' "
              "    kind: 'ram' "
              "  } "
              "  quantity: 15 "
              "} ");
          return Status::OK();
        }));

    loader_3_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*loader_3_, EstimateResources(_))
        .WillByDefault(Invoke([](ResourceAllocation* estimate) {
          *estimate = CreateProto<ResourceAllocation>(
              "resource_quantities { "
              "  resource { "
              "    device: 'gpu' "
              "    kind: 'ram' "
              "  } "
              "  quantity: 12 "
              "} ");
          return Status::OK();
        }));

    invalid_resources_loader_.reset(new NiceMock<test_util::MockLoader>);
    ON_CALL(*invalid_resources_loader_, EstimateResources(_))
        .WillByDefault(Invoke([](ResourceAllocation* estimate) {
          *estimate = CreateProto<ResourceAllocation>(
              "resource_quantities { "
              "  resource { "
              "    device: 'bogus_device' "
              "    device_instance { value: 0 } "
              "    kind: 'ram' "
              "  } "
              "  quantity: 4 "
              "} ");
          return Status::OK();
        }));

    // Disallow calls to Load()/Unload().
    for (auto* loader : {loader_0_.get(), loader_1_.get(), loader_2_.get(),
                         loader_3_.get(), invalid_resources_loader_.get()}) {
      EXPECT_CALL(*loader, Load(_)).Times(0);
      EXPECT_CALL(*loader, Unload()).Times(0);
    }
  }

  // The original total-resources valued provided to 'tracker_'.
  const ResourceAllocation total_resources_;

  // The object under testing.
  std::unique_ptr<ResourceTracker> tracker_;

  // Some mock loaders with specific resource estimates (see the constructor).
  std::unique_ptr<test_util::MockLoader> loader_0_;
  std::unique_ptr<test_util::MockLoader> loader_1_;
  std::unique_ptr<test_util::MockLoader> loader_2_;
  std::unique_ptr<test_util::MockLoader> loader_3_;
  std::unique_ptr<test_util::MockLoader> invalid_resources_loader_;
};

TEST_F(ResourceTrackerTest, UnboundTotalResources) {
  std::unique_ptr<ResourceUtil> util(
      new ResourceUtil({{{"main", 1}, {"gpu", 2}}}));
  std::unique_ptr<ResourceTracker> tracker;
  const auto unbound_resources = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 12 "
      "} ");
  EXPECT_FALSE(ResourceTracker::Create(
                   unbound_resources,
                   std::unique_ptr<ResourceUtil>(std::move(util)), &tracker)
                   .ok());
}

TEST_F(ResourceTrackerTest, UnnormalizedTotalResources) {
  std::unique_ptr<ResourceUtil> util(
      new ResourceUtil({{{"main", 1}, {"gpu", 2}}}));
  std::unique_ptr<ResourceTracker> tracker;
  const auto unnormalized_resources = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'main' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 12 "
      "} ");
  TF_ASSERT_OK(ResourceTracker::Create(
      unnormalized_resources, std::unique_ptr<ResourceUtil>(std::move(util)),
      &tracker));
  // The total_resources proto should get normalized.
  EXPECT_THAT(tracker->total_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 12 "
                          "} "));
}

TEST_F(ResourceTrackerTest, RecomputeUsedResources) {
  // Verify the initial state.
  EXPECT_THAT(tracker_->used_resources(), EqualsProto(""));
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));

  // Recompute used resources for {loader_0_, loader_1_, loader_3_}.
  TF_ASSERT_OK(tracker_->RecomputeUsedResources(
      {loader_0_.get(), loader_1_.get(), loader_3_.get()}));
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 12 "
                          "} "));
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));

  // Recompute used resources for just {loader_0_}.
  TF_ASSERT_OK(tracker_->RecomputeUsedResources({loader_0_.get()}));
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesSuccessWithUsedResourcesBound) {
  TF_ASSERT_OK(tracker_->RecomputeUsedResources({loader_0_.get()}));
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  bool success;
  TF_ASSERT_OK(tracker_->ReserveResources(*loader_2_, &success));
  EXPECT_TRUE(success);
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesFailureWithUsedResourcesBound) {
  TF_ASSERT_OK(tracker_->RecomputeUsedResources({loader_1_.get()}));
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  bool success;
  TF_ASSERT_OK(tracker_->ReserveResources(*loader_2_, &success));
  EXPECT_FALSE(success);
  // The used resources should remain unchanged (i.e. only reflect loader_1_).
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesSuccessWithUsedResourcesUnbound) {
  TF_ASSERT_OK(tracker_->RecomputeUsedResources({loader_3_.get()}));
  EXPECT_THAT(tracker_->used_resources(), EqualsProto("resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'gpu' "
                                                      "    kind: 'ram' "
                                                      "  } "
                                                      "  quantity: 12 "
                                                      "} "));

  // If just loader_3_ is loaded, loader_0_ should also fit.
  bool success;
  TF_ASSERT_OK(tracker_->ReserveResources(*loader_0_, &success));
  EXPECT_TRUE(success);
  EXPECT_THAT(tracker_->used_resources(),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 12 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'main' "
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
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, ReserveResourcesFailureWithUsedResourcesUnbound) {
  TF_ASSERT_OK(tracker_->RecomputeUsedResources({loader_3_.get()}));
  EXPECT_THAT(tracker_->used_resources(), EqualsProto("resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'gpu' "
                                                      "    kind: 'ram' "
                                                      "  } "
                                                      "  quantity: 12 "
                                                      "} "));

  // If loader_3_ is loaded, there isn't room for loader_1_, or for another
  // copy of loader_3_.
  bool success;
  TF_ASSERT_OK(tracker_->ReserveResources(*loader_1_, &success));
  EXPECT_FALSE(success);
  TF_ASSERT_OK(tracker_->ReserveResources(*loader_3_, &success));
  EXPECT_FALSE(success);
  // The used resources should remain unchanged (i.e. only reflect a single copy
  // of loader_3_).
  EXPECT_THAT(tracker_->used_resources(), EqualsProto("resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'gpu' "
                                                      "    kind: 'ram' "
                                                      "  } "
                                                      "  quantity: 12 "
                                                      "} "));
  EXPECT_THAT(tracker_->total_resources(), EqualsProto(total_resources_));
}

TEST_F(ResourceTrackerTest, InvalidResourceEstimate) {
  bool success;
  EXPECT_FALSE(
      tracker_->ReserveResources(*invalid_resources_loader_, &success).ok());
  EXPECT_FALSE(tracker_
                   ->RecomputeUsedResources(
                       {loader_0_.get(), invalid_resources_loader_.get()})
                   .ok());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
