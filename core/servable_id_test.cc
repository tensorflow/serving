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

#include "tensorflow_serving/core/servable_id.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

// Note: these tests use EXPECT_TRUE/FALSE in conjunction with the specific
// comparison operator being tested (e.g. ==), instead of e.g. EXPECT_EQ, to
// ensure that the exact operator targeted for testing is being invoked.

TEST(ServableIdTest, Equals) {
  EXPECT_TRUE((ServableId{"a", 1} == ServableId{"a", 1}));

  EXPECT_FALSE((ServableId{"b", 2} == ServableId{"a", 2}));
  EXPECT_FALSE((ServableId{"b", 1} == ServableId{"b", 2}));
}

TEST(ServableIdTest, NotEquals) {
  EXPECT_FALSE((ServableId{"a", 1} != ServableId{"a", 1}));

  EXPECT_TRUE((ServableId{"b", 2} != ServableId{"a", 2}));
  EXPECT_TRUE((ServableId{"b", 1} != ServableId{"b", 2}));
}

TEST(ServableIdTest, LessThan) {
  EXPECT_TRUE((ServableId{"a", 1} < ServableId{"b", 1}));
  EXPECT_TRUE((ServableId{"b", 1} < ServableId{"b", 2}));
  EXPECT_TRUE((ServableId{"a", 1} < ServableId{"b", 2}));

  EXPECT_FALSE((ServableId{"a", 1} < ServableId{"a", 1}));

  EXPECT_FALSE((ServableId{"b", 1} < ServableId{"a", 1}));
  EXPECT_FALSE((ServableId{"b", 2} < ServableId{"b", 1}));
  EXPECT_FALSE((ServableId{"b", 2} < ServableId{"a", 1}));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
