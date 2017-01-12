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

#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/core/test_util/mock_storage_path_target.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::StrictMock;
using ::testing::Return;

namespace tensorflow {
namespace serving {

namespace internal {

class FileSystemStoragePathSourceTestAccess {
 public:
  // Assumes 'source' is a FileSystemStoragePathSource.
  explicit FileSystemStoragePathSourceTestAccess(Source<StoragePath>* source)
      : source_(static_cast<FileSystemStoragePathSource*>(source)) {}

  Status PollFileSystemAndInvokeCallback() {
    return source_->PollFileSystemAndInvokeCallback();
  }

 private:
  FileSystemStoragePathSource* source_;

  TF_DISALLOW_COPY_AND_ASSIGN(FileSystemStoragePathSourceTestAccess);
};

}  // namespace internal

namespace {

TEST(FileSystemStoragePathSourceTest, NoVersionsAtStartup) {
  for (bool base_path_exists : {false, true}) {
    const string base_path = io::JoinPath(
        testing::TmpDir(),
        strings::StrCat("NoVersionsAtStartup",
                        base_path_exists ? "" : "_nonexistent_base_path"));
    if (base_path_exists) {
      TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
      TF_ASSERT_OK(Env::Default()->CreateDir(
          io::JoinPath(base_path, "non_numerical_child")));
    }

    for (bool fail_if_zero_versions_at_startup : {false, true}) {
      auto config = test_util::CreateProto<FileSystemStoragePathSourceConfig>(
          strings::Printf("servable_name: 'test_servable_name' "
                          "base_path: '%s' "
                          "fail_if_zero_versions_at_startup: %s "
                          // Disable the polling thread.
                          "file_system_poll_wait_seconds: -1 ",
                          base_path.c_str(),
                          fail_if_zero_versions_at_startup ? "true" : "false"));
      std::unique_ptr<FileSystemStoragePathSource> source;
      bool success = FileSystemStoragePathSource::Create(config, &source).ok();
      EXPECT_EQ(!fail_if_zero_versions_at_startup, success);
      if (success) {
        std::unique_ptr<test_util::MockStoragePathTarget> target(
            new StrictMock<test_util::MockStoragePathTarget>);
        ConnectSourceToTarget(source.get(), target.get());
        if (base_path_exists) {
          // Poll and expect zero aspired versions.
          EXPECT_CALL(*target,
                      SetAspiredVersions(Eq("test_servable_name"), IsEmpty()));
          TF_ASSERT_OK(
              internal::FileSystemStoragePathSourceTestAccess(source.get())
                  .PollFileSystemAndInvokeCallback());
        } else {
          // Poll and expect an error.
          EXPECT_FALSE(
              internal::FileSystemStoragePathSourceTestAccess(source.get())
                  .PollFileSystemAndInvokeCallback()
                  .ok());
        }
      }
    }
  }
}

TEST(FileSystemStoragePathSourceTest, FilesAppearAfterStartup) {
  const string base_path =
      io::JoinPath(testing::TmpDir(), "FilesAppearAfterStartup");

  auto config = test_util::CreateProto<FileSystemStoragePathSourceConfig>(
      strings::Printf("servable_name: 'test_servable_name' "
                      "base_path: '%s' "
                      "fail_if_zero_versions_at_startup: false "
                      // Disable the polling thread.
                      "file_system_poll_wait_seconds: -1 ",
                      base_path.c_str()));
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  // Poll and expect an error.
  EXPECT_FALSE(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback()
                   .ok());

  // Inject the base-path and version, and re-poll.
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "3")));
  EXPECT_CALL(*target, SetAspiredVersions(Eq("test_servable_name"),
                                          ElementsAre(ServableData<StoragePath>(
                                              {"test_servable_name", 3},
                                              io::JoinPath(base_path, "3")))));
  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

TEST(FileSystemStoragePathSourceTest, MultipleVersions) {
  const string base_path = io::JoinPath(testing::TmpDir(), "MultipleVersions");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
  TF_ASSERT_OK(Env::Default()->CreateDir(
      io::JoinPath(base_path, "non_numerical_child")));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "42")));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "17")));

  auto config = test_util::CreateProto<FileSystemStoragePathSourceConfig>(
      strings::Printf("servable_name: 'test_servable_name' "
                      "base_path: '%s' "
                      // Disable the polling thread.
                      "file_system_poll_wait_seconds: -1 ",
                      base_path.c_str()));
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  EXPECT_CALL(*target, SetAspiredVersions(Eq("test_servable_name"),
                                          ElementsAre(ServableData<StoragePath>(
                                              {"test_servable_name", 42},
                                              io::JoinPath(base_path, "42")))));
  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

TEST(FileSystemStoragePathSourceTest, MultipleVersionsAtTheSameTime) {
  const string base_path =
      io::JoinPath(testing::TmpDir(), "MultipleVersionsAtTheSameTime");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
  TF_ASSERT_OK(Env::Default()->CreateDir(
      io::JoinPath(base_path, "non_numerical_child")));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "42")));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "17")));

  auto config = test_util::CreateProto<FileSystemStoragePathSourceConfig>(
      strings::Printf("servables: { "
                      "  version_policy: ALL_VERSIONS "
                      "  servable_name: 'test_servable_name' "
                      "  base_path: '%s' "
                      "} "
                      // Disable the polling thread.
                      "file_system_poll_wait_seconds: -1 ",
                      base_path.c_str()));
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  EXPECT_CALL(
      *target,
      SetAspiredVersions(
          Eq("test_servable_name"),
          ElementsAre(
              ServableData<StoragePath>({"test_servable_name", 17},
                                        io::JoinPath(base_path, "17")),
              ServableData<StoragePath>({"test_servable_name", 42},
                                        io::JoinPath(base_path, "42")))));

  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

TEST(FileSystemStoragePathSourceTest, MultipleServables) {
  FileSystemStoragePathSourceConfig config;
  config.set_fail_if_zero_versions_at_startup(false);
  config.set_file_system_poll_wait_seconds(-1);  // Disable the polling thread.

  // Servable 0 has two versions.
  const string base_path_0 =
      io::JoinPath(testing::TmpDir(), "MultipleServables_0");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path_0));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path_0, "1")));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path_0, "3")));
  auto* servable_0 = config.add_servables();
  servable_0->set_servable_name("servable_0");
  servable_0->set_base_path(base_path_0);

  // Servable 1 has one version.
  const string base_path_1 =
      io::JoinPath(testing::TmpDir(), "MultipleServables_1");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path_1));
  TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path_1, "42")));
  auto* servable_1 = config.add_servables();
  servable_1->set_servable_name("servable_1");
  servable_1->set_base_path(base_path_1);

  // Servable 2 has no versions.
  const string base_path_2 =
      io::JoinPath(testing::TmpDir(), "MultipleServables_2");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path_2));
  auto* servable_2 = config.add_servables();
  servable_2->set_servable_name("servable_2");
  servable_2->set_base_path(base_path_2);

  // Create a source and connect it to a mock target.
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  // Have the source poll the FS, and expect certain callback calls.
  EXPECT_CALL(*target,
              SetAspiredVersions(
                  Eq("servable_0"),
                  ElementsAre(ServableData<StoragePath>(
                      {"servable_0", 3}, io::JoinPath(base_path_0, "3")))));
  EXPECT_CALL(*target,
              SetAspiredVersions(
                  Eq("servable_1"),
                  ElementsAre(ServableData<StoragePath>(
                      {"servable_1", 42}, io::JoinPath(base_path_1, "42")))));
  EXPECT_CALL(*target, SetAspiredVersions(Eq("servable_2"), IsEmpty()));
  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

TEST(FileSystemStoragePathSourceTest, ChangeSetOfServables) {
  FileSystemStoragePathSourceConfig config;
  config.set_fail_if_zero_versions_at_startup(false);
  config.set_file_system_poll_wait_seconds(-1);  // Disable the polling thread.

  // Create three servables, each with a single version numbered 0.
  const string base_path_prefix =
      io::JoinPath(testing::TmpDir(), "ChangeSetOfServables_");
  for (int i = 0; i <= 2; ++i) {
    const string base_path = strings::StrCat(base_path_prefix, i);
    TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
    TF_ASSERT_OK(Env::Default()->CreateDir(io::JoinPath(base_path, "0")));
  }

  // Configure a source initially with servables 0 and 1.
  for (int i : {0, 1}) {
    auto* servable = config.add_servables();
    servable->set_servable_name(strings::StrCat("servable_", i));
    servable->set_base_path(strings::StrCat(base_path_prefix, i));
  }
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());
  for (int i : {0, 1}) {
    EXPECT_CALL(
        *target,
        SetAspiredVersions(
            Eq(strings::StrCat("servable_", i)),
            ElementsAre(ServableData<StoragePath>(
                {strings::StrCat("servable_", i), 0},
                io::JoinPath(strings::StrCat(base_path_prefix, i), "0")))));
  }
  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());

  // Reconfigure the source to have servables 1 and 2 (dropping servable 0).
  config.clear_servables();
  for (int i : {1, 2}) {
    auto* servable = config.add_servables();
    servable->set_servable_name(strings::StrCat("servable_", i));
    servable->set_base_path(strings::StrCat(base_path_prefix, i));
  }
  // Servable 0 should get a zero-versions callback, causing the manager to
  // unload it.
  EXPECT_CALL(*target, SetAspiredVersions(Eq("servable_0"), IsEmpty()));
  // Servables 1 and 2 should each get a one-version callback. Importantly,
  // servable 1 (which is in both the old and new configs) should *not* see a
  // zero-version callback followed by a one-version one, which could cause the
  // manager to temporarily unload the servable.
  for (int i : {1, 2}) {
    EXPECT_CALL(
        *target,
        SetAspiredVersions(
            Eq(strings::StrCat("servable_", i)),
            ElementsAre(ServableData<StoragePath>(
                {strings::StrCat("servable_", i), 0},
                io::JoinPath(strings::StrCat(base_path_prefix, i), "0")))));
  }
  TF_ASSERT_OK(source->UpdateConfig(config));
  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

TEST(FileSystemStoragePathSourceTest, AttemptToChangePollingPeriod) {
  FileSystemStoragePathSourceConfig config;
  config.set_file_system_poll_wait_seconds(1);
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  FileSystemStoragePathSourceConfig new_config = config;
  new_config.set_file_system_poll_wait_seconds(5);
  EXPECT_FALSE(source->UpdateConfig(new_config).ok());
}

TEST(FileSystemStoragePathSourceTest, ParseTimestampedVersion) {
  static_assert(static_cast<int32>(20170111173521LL) == 944751505,
                "Version overflows if cast to int32.");
  const string base_path =
      io::JoinPath(testing::TmpDir(), "ParseTimestampedVersion");
  TF_ASSERT_OK(Env::Default()->CreateDir(base_path));
  TF_ASSERT_OK(
      Env::Default()->CreateDir(io::JoinPath(base_path, "20170111173521")));
  auto config = test_util::CreateProto<FileSystemStoragePathSourceConfig>(
      strings::Printf("servables: { "
                      "  version_policy: ALL_VERSIONS "
                      "  servable_name: 'test_servable_name' "
                      "  base_path: '%s' "
                      "} "
                      // Disable the polling thread.
                      "file_system_poll_wait_seconds: -1 ",
                      base_path.c_str()));
  std::unique_ptr<FileSystemStoragePathSource> source;
  TF_ASSERT_OK(FileSystemStoragePathSource::Create(config, &source));
  std::unique_ptr<test_util::MockStoragePathTarget> target(
      new StrictMock<test_util::MockStoragePathTarget>);
  ConnectSourceToTarget(source.get(), target.get());

  EXPECT_CALL(*target, SetAspiredVersions(
                           Eq("test_servable_name"),
                           ElementsAre(ServableData<StoragePath>(
                               {"test_servable_name", 20170111173521LL},
                               io::JoinPath(base_path, "20170111173521")))));

  TF_ASSERT_OK(internal::FileSystemStoragePathSourceTestAccess(source.get())
                   .PollFileSystemAndInvokeCallback());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
