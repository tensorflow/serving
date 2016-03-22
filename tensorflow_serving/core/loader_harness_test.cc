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

#include "tensorflow_serving/core/loader_harness.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/any_ptr.h"

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::InvokeWithoutArgs;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;

namespace tensorflow {
namespace serving {

namespace {

void QuiesceAndUnload(LoaderHarness* const harness) {
  harness->StartQuiescing();
  harness->DoneQuiescing();
  harness->Unload();
}
}

TEST(LoaderHarnessTest, Init) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  EXPECT_EQ((ServableId{"test", 0}), harness.id());
  EXPECT_EQ(LoaderHarness::kNew, harness.state());
  EXPECT_TRUE(harness.is_aspired());
  EXPECT_EQ(harness.is_aspired(), harness.loader_state_snapshot().is_aspired);
  EXPECT_EQ(harness.state(), harness.loader_state_snapshot().state);
}

TEST(LoaderHarnessTest, set_is_aspired) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  harness.set_is_aspired(false);
  EXPECT_FALSE(harness.is_aspired());
}

TEST(LoaderHarnessTest, Quiesce) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  EXPECT_CALL(*loader, Unload()).WillOnce(Return());

  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  harness.StartQuiescing();
  EXPECT_EQ(LoaderHarness::State::kQuiescing, harness.state());

  harness.DoneQuiescing();
  EXPECT_EQ(LoaderHarness::State::kQuiesced, harness.state());

  // Otherwise we break the dtor invariant and check-fail.
  harness.Unload();
}

TEST(LoaderHarnessTest, Load) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Unload()).WillOnce(Return());

  const auto available_resources = test_util::CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'cpu' "
      "    kind: 'ram' "
      "  } "
      "  quantity: 16 "
      "} ");

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(test_util::EqualsProto(available_resources)))
      .WillOnce(InvokeWithoutArgs([&load_called, &load_should_return]() {
        load_called.Notify();
        load_should_return.WaitForNotification();
        return Status::OK();
      }));
  {
    std::unique_ptr<Thread> test_thread(Env::Default()->StartThread(
        ThreadOptions(), "test", [&harness, &available_resources]() {
          EXPECT_TRUE(harness.Load(available_resources).ok());
        }));
    load_called.WaitForNotification();
    EXPECT_EQ(LoaderHarness::kLoading, harness.state());
    load_should_return.Notify();
    // Deleting the thread here forces join and ensures that
    // LoaderHarness::Load() returns.
  }
  EXPECT_EQ(LoaderHarness::kReady, harness.state());

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, Unload) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  EXPECT_TRUE(harness.Load(ResourceAllocation()).ok());

  Notification unload_called;
  Notification unload_should_return;
  EXPECT_CALL(*loader, Unload())
      .WillOnce(InvokeWithoutArgs([&unload_called, &unload_should_return]() {
        unload_called.Notify();
        unload_should_return.WaitForNotification();
      }));
  {
    std::unique_ptr<Thread> test_thread(Env::Default()->StartThread(
        ThreadOptions(), "test", [&harness]() { QuiesceAndUnload(&harness); }));
    unload_called.WaitForNotification();
    EXPECT_EQ(LoaderHarness::kUnloading, harness.state());
    unload_should_return.Notify();
    // Deleting the thread here forces join and ensures that
    // LoaderHarness::Unload() returns.
  }
  EXPECT_EQ(LoaderHarness::kDisabled, harness.state());
}

TEST(ServableVersionHarnessTest, LoadError) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs([&load_called, &load_should_return]() {
        load_called.Notify();
        load_should_return.WaitForNotification();
        return Status(error::UNKNOWN, "test load error");
      }));
  {
    std::unique_ptr<Thread> test_thread(
        Env::Default()->StartThread(ThreadOptions(), "test", [&harness]() {
          Status status = harness.Load(ResourceAllocation());
          EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
        }));
    load_called.WaitForNotification();
    EXPECT_EQ(LoaderHarness::kLoading, harness.state());
    load_should_return.Notify();
  }
  EXPECT_EQ(LoaderHarness::kError, harness.state());
}

TEST(ServableVersionHarnessTest, RetryOnLoadErrorFinallySucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(
      ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
      {2 /* max_num_load_tries */, 1 /* load_retry_interval_micros */});

  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs(
          []() { return Status(error::UNKNOWN, "test load error"); }))
      .WillOnce(InvokeWithoutArgs([]() { return Status::OK(); }));
  const Status status = harness.Load(ResourceAllocation());
  TF_EXPECT_OK(status);

  QuiesceAndUnload(&harness);
}

// Tests cancelling a load by setting is_aspired to false,
TEST(ServableVersionHarnessTest, RetryOnLoadErrorCancelledLoad) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        {10 /* max_num_load_tries */, -1});

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs([&load_called, &load_should_return]() {
        load_called.Notify();
        load_should_return.WaitForNotification();
        return Status(error::UNKNOWN, "test load error");
      }))
      .WillRepeatedly(InvokeWithoutArgs([]() { return Status::OK(); }));
  std::unique_ptr<Thread> test_thread(
      Env::Default()->StartThread(ThreadOptions(), "test", [&harness]() {
        const Status status = harness.Load(ResourceAllocation());
        EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
      }));
  load_called.WaitForNotification();
  harness.set_is_aspired(false);
  load_should_return.Notify();
}

TEST(ServableVersionHarnessTest, RetryOnLoadErrorFinallyFails) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        {2 /* max_num_load_tries */, -1});

  EXPECT_CALL(*loader, Load(_))
      .Times(2)
      .WillRepeatedly(InvokeWithoutArgs(
          []() { return Status(error::UNKNOWN, "test load error"); }));
  const Status status = harness.Load(ResourceAllocation());
  EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
}

TEST(LoaderHarnessTest, ExternallySignalledError) {
  LoaderHarness harness(ServableId{"test", 0}, nullptr);
  EXPECT_EQ(LoaderHarness::State::kNew, harness.state());
  const Status status = Status(error::UNKNOWN, "Some unknown error");
  harness.Error(status);
  EXPECT_EQ(LoaderHarness::State::kError, harness.state());
  EXPECT_EQ(status, harness.status());
}

}  // namespace serving
}  // namespace tensorflow
