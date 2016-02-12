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

#include "tensorflow_serving/core/source_adapter.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/test_util/mock_storage_path_target.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Return;
using ::testing::StrictMock;

namespace tensorflow {
namespace serving {
namespace {

// A SourceAdapter that expects all aspired-versions requests to be empty.
class LimitedAdapter : public SourceAdapter<StoragePath, StoragePath> {
 public:
  LimitedAdapter() = default;
  ~LimitedAdapter() override = default;

 protected:
  std::vector<ServableData<StoragePath>> Adapt(
      const StringPiece servable_name,
      std::vector<ServableData<StoragePath>> versions) override {
    CHECK(versions.empty());
    return {};
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LimitedAdapter);
};

TEST(SourceAdapterTest, SetAspiredVersionsBlocksUntilTargetConnected) {
  LimitedAdapter adapter;
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  std::unique_ptr<Thread> connect_target(Env::Default()->StartThread(
      {}, "ConnectTarget",
      [&adapter, &target] {
        // Sleep for a long time before connecting the target, to make it very
        // likely that SetAspiredVersions() gets called first and has to block.
        Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /* 1 second */);
        ConnectSourceToTarget(&adapter, target.get());
      }));
  EXPECT_CALL(*target, SetAspiredVersions(Eq("foo"), IsEmpty()));
  adapter.SetAspiredVersions("foo", {});
}

// A UnarySourceAdapter that appends "_adapted" to every incoming StoragePath.
class TestUnaryAdapter : public UnarySourceAdapter<StoragePath, StoragePath> {
 public:
  TestUnaryAdapter() = default;
  ~TestUnaryAdapter() override = default;

 protected:
  Status Convert(const StoragePath& data,
                 StoragePath* converted_data) override {
    if (data == "invalid") {
      return errors::InvalidArgument(
          "TestUnaryAdapter Convert() dutifully failing on \"invalid\" data");
    }
    *converted_data = strings::StrCat(data, "_adapted");
    return Status::OK();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TestUnaryAdapter);
};

TEST(UnarySourceAdapterTest, Basic) {
  TestUnaryAdapter adapter;
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(&adapter, target.get());
  EXPECT_CALL(
      *target,
      SetAspiredVersions(
          Eq("foo"),
          ElementsAre(
              ServableData<StoragePath>({"foo", 0}, "mrop_adapted"),
              ServableData<StoragePath>(
                  {"foo", 1}, errors::InvalidArgument(
                                  "TestUnaryAdapter Convert() dutifully "
                                  "failing on \"invalid\" data")),
              ServableData<StoragePath>({"foo", 2}, errors::Unknown("d'oh")))));
  adapter.SetAspiredVersions(
      "foo", {ServableData<StoragePath>({"foo", 0}, "mrop"),
              ServableData<StoragePath>({"foo", 1}, "invalid"),
              ServableData<StoragePath>({"foo", 2}, errors::Unknown("d'oh"))});
}

TEST(ErrorInjectingSourceAdapterTest, Basic) {
  ErrorInjectingSourceAdapter<string, string> adapter(
      errors::Unknown("Injected error"));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(&adapter, target.get());
  EXPECT_CALL(
      *target,
      SetAspiredVersions(
          Eq("foo"),
          ElementsAre(ServableData<StoragePath>(
                          {"foo", 0}, errors::Unknown("Injected error")),
                      ServableData<StoragePath>(
                          {"foo", 1}, errors::Unknown("Original error")))));
  adapter.SetAspiredVersions(
      "foo", {ServableData<StoragePath>({"foo", 0}, "mrop"),
              ServableData<StoragePath>({"foo", 1},
                                        errors::Unknown("Original error"))});
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
