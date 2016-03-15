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

#include "tensorflow_serving/resources/resource_util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/test_util/test_util.h"

using ::tensorflow::serving::test_util::CreateProto;
using ::tensorflow::serving::test_util::EqualsProto;

namespace tensorflow {
namespace serving {
namespace {

class ResourceUtilTest : public ::testing::Test {
 protected:
  ResourceUtilTest() : util_({{{"cpu", 1}, {"gpu", 2}}}) {}

  // The object under testing.
  ResourceUtil util_;
};

TEST_F(ResourceUtilTest, VerifyValidity) {
  // Empty.
  TF_EXPECT_OK(util_.VerifyValidity(CreateProto<ResourceAllocation>("")));

  // Unbound.
  TF_EXPECT_OK(util_.VerifyValidity(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
  TF_EXPECT_OK(util_.VerifyValidity(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} ")));

  // Bound to a valid instance.
  TF_EXPECT_OK(util_.VerifyValidity(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} ")));
  TF_EXPECT_OK(util_.VerifyValidity(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 1 } "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} ")));

  // Non-existent device.
  EXPECT_FALSE(util_
                   .VerifyValidity(CreateProto<ResourceAllocation>(
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'nonexistent_device' "
                       "    kind: 'processing' "
                       "  } "
                       "  quantity: 100 "
                       "} "))
                   .ok());
  EXPECT_FALSE(util_
                   .VerifyValidity(CreateProto<ResourceAllocation>(
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'cpu' "
                       "    kind: 'processing' "
                       "  } "
                       "  quantity: 100 "
                       "} "
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'nonexistent_device' "
                       "    kind: 'ram' "
                       "  } "
                       "  quantity: 4 "
                       "} "))
                   .ok());

  // Bound to an invalid instance.
  EXPECT_FALSE(util_
                   .VerifyValidity(CreateProto<ResourceAllocation>(
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'gpu' "
                       "    device_instance { value: 2 } "
                       "    kind: 'ram' "
                       "  } "
                       "  quantity: 4 "
                       "} "))
                   .ok());

  // Repeated entries for the same resource.
  EXPECT_FALSE(util_
                   .VerifyValidity(
                       CreateProto<ResourceAllocation>("resource_quantities { "
                                                       "  resource { "
                                                       "    device: 'gpu' "
                                                       "    kind: 'ram' "
                                                       "  } "
                                                       "  quantity: 2 "
                                                       "} "
                                                       "resource_quantities { "
                                                       "  resource { "
                                                       "    device: 'gpu' "
                                                       "    kind: 'ram' "
                                                       "  } "
                                                       "  quantity: 4 "
                                                       "} "))
                   .ok());
  EXPECT_FALSE(util_
                   .VerifyValidity(CreateProto<ResourceAllocation>(
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'gpu' "
                       "    device_instance { value: 0 } "
                       "    kind: 'ram' "
                       "  } "
                       "  quantity: 2 "
                       "} "
                       "resource_quantities { "
                       "  resource { "
                       "    device: 'gpu' "
                       "    device_instance { value: 0 } "
                       "    kind: 'ram' "
                       "  } "
                       "  quantity: 4 "
                       "} "))
                   .ok());
}

TEST_F(ResourceUtilTest, Normalize) {
  EXPECT_THAT(util_.Normalize(CreateProto<ResourceAllocation>("")),
              EqualsProto(""));
  EXPECT_THAT(
      util_.Normalize(CreateProto<ResourceAllocation>("resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'gpu' "
                                                      "    kind: 'processing' "
                                                      "  } "
                                                      "  quantity: 100 "
                                                      "} "
                                                      "resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'gpu' "
                                                      "    kind: 'ram' "
                                                      "  } "
                                                      "  quantity: 0 "
                                                      "} ")),
      EqualsProto("resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    kind: 'processing' "
                  "  } "
                  "  quantity: 100 "
                  "} "));
  EXPECT_THAT(
      util_.Normalize(CreateProto<ResourceAllocation>("resource_quantities { "
                                                      "  resource { "
                                                      "    device: 'cpu' "
                                                      "    kind: 'ram' "
                                                      "  } "
                                                      "  quantity: 2 "
                                                      "} ")),
      EqualsProto("resource_quantities { "
                  "  resource { "
                  "    device: 'cpu' "
                  "    device_instance { value: 0 } "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 2 "
                  "} "));
  // No-op.
  EXPECT_THAT(util_.Normalize(CreateProto<ResourceAllocation>(
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'cpu' "
                  "    device_instance { value: 0 } "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 2 "
                  "} ")),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 2 "
                          "} "));
}

TEST_F(ResourceUtilTest, IsNormalized) {
  EXPECT_TRUE(util_.IsNormalized(CreateProto<ResourceAllocation>("")));
  EXPECT_TRUE(util_.IsNormalized(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
  EXPECT_TRUE(util_.IsNormalized(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
  EXPECT_TRUE(util_.IsNormalized(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));

  EXPECT_FALSE(util_.IsNormalized(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 0 "
                                      "} ")));
  EXPECT_FALSE(util_.IsNormalized(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
}

TEST_F(ResourceUtilTest, IsBound) {
  EXPECT_TRUE(util_.IsBound(CreateProto<ResourceAllocation>("")));
  EXPECT_TRUE(util_.IsBound(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
  EXPECT_TRUE(util_.IsBound(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 1 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));

  EXPECT_FALSE(
      util_.IsBound(CreateProto<ResourceAllocation>("resource_quantities { "
                                                    "  resource { "
                                                    "    device: 'gpu' "
                                                    "    kind: 'processing' "
                                                    "  } "
                                                    "  quantity: 100 "
                                                    "} ")));
  EXPECT_FALSE(util_.IsBound(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} ")));
}

TEST_F(ResourceUtilTest, AddEmpty) {
  auto base = CreateProto<ResourceAllocation>("");
  const auto to_add = CreateProto<ResourceAllocation>("");
  util_.Add(to_add, &base);
  EXPECT_THAT(base, EqualsProto(""));
}

TEST_F(ResourceUtilTest, AddBasic) {
  auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 100 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 8 "
      "} ");
  const auto to_add = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 300 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  util_.Add(to_add, &base);
  EXPECT_THAT(base, EqualsProto("resource_quantities { "
                                "  resource { "
                                "    device: 'cpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'processing' "
                                "  } "
                                "  quantity: 400 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'cpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 8 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 4 "
                                "} "));
}

TEST_F(ResourceUtilTest, AddBoundAndUnbound) {
  auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 8 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  const auto to_add = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
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
      "  quantity: 2 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 12 "
      "} ");
  util_.Add(to_add, &base);
  EXPECT_THAT(base, EqualsProto("resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 24 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 6 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    device_instance { value: 1 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 12 "
                                "} "));
}

TEST_F(ResourceUtilTest, SubtractEmpty) {
  auto base = CreateProto<ResourceAllocation>("");
  const auto to_subtract = CreateProto<ResourceAllocation>("");
  EXPECT_TRUE(util_.Subtract(to_subtract, &base));
  EXPECT_THAT(base, EqualsProto(""));
}

TEST_F(ResourceUtilTest, SubtractBasic) {
  auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 300 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 8 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 16 "
      "} ");
  const auto to_subtract = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 100 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  EXPECT_TRUE(util_.Subtract(to_subtract, &base));
  EXPECT_THAT(base, EqualsProto("resource_quantities { "
                                "  resource { "
                                "    device: 'cpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'processing' "
                                "  } "
                                "  quantity: 200 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'cpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 4 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 16 "
                                "} "));
}

TEST_F(ResourceUtilTest, SubtractNegativeResult) {
  const auto original_base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 100 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 16 "
      "} ");
  ResourceAllocation base = original_base;
  const auto to_subtract = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 50 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 6 "
      "} ");
  EXPECT_FALSE(util_.Subtract(to_subtract, &base));
  // Upon detecting a negative result, it should leave 'base' unchanged.
  EXPECT_THAT(base, EqualsProto(original_base));
}

TEST_F(ResourceUtilTest, SubtractNormalizeOutput) {
  auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 10 "
      "} ");
  const auto to_subtract = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 10 "
      "} ");
  EXPECT_TRUE(util_.Subtract(to_subtract, &base));
  EXPECT_THAT(base, EqualsProto(""));
}

TEST_F(ResourceUtilTest, SubtractBoundAndUnbound) {
  auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
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
      "  quantity: 4 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 12 "
      "} ");
  const auto to_subtract = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 8 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 2 "
      "} ");
  EXPECT_TRUE(util_.Subtract(to_subtract, &base));
  EXPECT_THAT(base, EqualsProto("resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 8 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    device_instance { value: 0 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 2 "
                                "} "
                                "resource_quantities { "
                                "  resource { "
                                "    device: 'gpu' "
                                "    device_instance { value: 1 } "
                                "    kind: 'ram' "
                                "  } "
                                "  quantity: 12 "
                                "} "));
}

TEST_F(ResourceUtilTest, LessThanOrEqualEmpty) {
  const auto a = CreateProto<ResourceAllocation>("");
  EXPECT_TRUE(util_.LessThanOrEqual(a, a));
}

TEST_F(ResourceUtilTest, LessThanOrEqualOneEntry) {
  const auto a = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 10 "
      "} ");
  const auto b = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 20 "
      "} ");
  EXPECT_TRUE(util_.LessThanOrEqual(a, a));
  EXPECT_TRUE(util_.LessThanOrEqual(a, b));
  EXPECT_FALSE(util_.LessThanOrEqual(b, a));
}

TEST_F(ResourceUtilTest, LessThanOrEqualTwoEntries) {
  const auto a = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 10 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  const auto b = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 5 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  EXPECT_TRUE(util_.LessThanOrEqual(a, a));
  EXPECT_TRUE(util_.LessThanOrEqual(b, a));
  EXPECT_FALSE(util_.LessThanOrEqual(a, b));
}

TEST_F(ResourceUtilTest, LessThanOrEqualImplicitZero) {
  const auto a = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 10 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");
  const auto b = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 5 "
      "} ");
  EXPECT_TRUE(util_.LessThanOrEqual(b, a));
  EXPECT_FALSE(util_.LessThanOrEqual(a, b));
}

// Test LessThanOrEqual(lhs, rhs) where 'lhs' is unbound. ('rhs' is always
// bound.)
TEST_F(ResourceUtilTest, LessThanOrEqualUnbound) {
  const auto base = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 1000 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 100 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'processing' "
      "  } "
      "  quantity: 50 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 0 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 2 "
      "} "
      "resource_quantities { "
      "  resource { "
      "    device: 'gpu' "
      "    device_instance { value: 1 } "
      "    kind: 'ram' "
      "  } "
      "  quantity: 4 "
      "} ");

  EXPECT_TRUE(util_.LessThanOrEqual(CreateProto<ResourceAllocation>(""), base));
  EXPECT_TRUE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'cpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "),
      base));
  EXPECT_TRUE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "),
      base));
  EXPECT_FALSE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 5 "
                                      "} "),
      base));
  EXPECT_TRUE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "),
      base));
  EXPECT_TRUE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 50 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "),
      base));
  EXPECT_TRUE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 90 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 1 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 40 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 10 "
                                      "} "),
      base));
  EXPECT_FALSE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 101 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 4 "
                                      "} "),
      base));
  EXPECT_FALSE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 100 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'ram' "
                                      "  } "
                                      "  quantity: 5 "
                                      "} "),
      base));
  EXPECT_FALSE(util_.LessThanOrEqual(
      CreateProto<ResourceAllocation>("resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 0 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 90 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    device_instance { value: 1 } "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 40 "
                                      "} "
                                      "resource_quantities { "
                                      "  resource { "
                                      "    device: 'gpu' "
                                      "    kind: 'processing' "
                                      "  } "
                                      "  quantity: 20 "
                                      "} "),
      base));
}

TEST_F(ResourceUtilTest, Overbind) {
  EXPECT_THAT(util_.Overbind(CreateProto<ResourceAllocation>("")),
              EqualsProto(""));
  EXPECT_THAT(util_.Overbind(CreateProto<ResourceAllocation>(
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'cpu' "
                  "    device_instance { value: 0 } "
                  "    kind: 'processing' "
                  "  } "
                  "  quantity: 100 "
                  "} "
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    device_instance { value: 0 } "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 4 "
                  "} ")),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'cpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'processing' "
                          "  } "
                          "  quantity: 100 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 4 "
                          "} "));
  EXPECT_THAT(util_.Overbind(CreateProto<ResourceAllocation>(
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 4 "
                  "} "
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    device_instance { value: 1 } "
                  "    kind: 'processing' "
                  "  } "
                  "  quantity: 100 "
                  "} ")),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 4 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 1 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 4 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 1 } "
                          "    kind: 'processing' "
                          "  } "
                          "  quantity: 100 "
                          "} "));
  EXPECT_THAT(util_.Overbind(CreateProto<ResourceAllocation>(
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 4 "
                  "} "
                  "resource_quantities { "
                  "  resource { "
                  "    device: 'gpu' "
                  "    device_instance { value: 1 } "
                  "    kind: 'ram' "
                  "  } "
                  "  quantity: 2 "
                  "} ")),
              EqualsProto("resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 0 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 4 "
                          "} "
                          "resource_quantities { "
                          "  resource { "
                          "    device: 'gpu' "
                          "    device_instance { value: 1 } "
                          "    kind: 'ram' "
                          "  } "
                          "  quantity: 6 "
                          "} "));
}

TEST_F(ResourceUtilTest, ResourceEquality) {
  std::vector<Resource> distinct_protos = {
      CreateProto<Resource>(""), CreateProto<Resource>("device: 'cpu' "
                                                       "kind: 'ram' "),
      CreateProto<Resource>("device: 'gpu' "
                            "kind: 'ram' "),
      CreateProto<Resource>("device: 'gpu' "
                            "kind: 'processing' "),
      CreateProto<Resource>("device: 'gpu' "
                            " device_instance { value: 1 } "
                            "kind: 'processing' ")};
  for (int i = 0; i < distinct_protos.size(); ++i) {
    for (int j = 0; j < distinct_protos.size(); ++j) {
      EXPECT_EQ(i == j, operator==(distinct_protos[i], distinct_protos[j]));
    }
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
