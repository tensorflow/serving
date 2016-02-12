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

#include "tensorflow_serving/core/servable_data.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

TEST(ServableDataTest, NoError) {
  ServableId id = {"name", 42};
  ServableData<string> data(id, "yo");
  EXPECT_EQ(id, data.id());
  TF_EXPECT_OK(data.status());
  EXPECT_EQ("yo", data.DataOrDie());
  EXPECT_EQ("yo", data.ConsumeDataOrDie());
}

TEST(ServableDataTest, StaticCreateNoError) {
  ServableId id = {"name", 42};
  auto data = CreateServableData(id, "yo");
  EXPECT_EQ(id, data.id());
  TF_EXPECT_OK(data.status());
  EXPECT_EQ("yo", data.DataOrDie());
  EXPECT_EQ("yo", data.ConsumeDataOrDie());
}

TEST(ServableDataTest, Error) {
  ServableId id = {"name", 42};
  ServableData<string> data(id, errors::Unknown("d'oh"));
  EXPECT_EQ(id, data.id());
  EXPECT_EQ(errors::Unknown("d'oh"), data.status());
}

}  // namespace serving
}  // namespace tensorflow
