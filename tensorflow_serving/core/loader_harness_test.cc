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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::InvokeWithoutArgs;
using ::testing::InSequence;
using ::testing::IsNull;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrictMock;
using test_util::EqualsProto;

void QuiesceAndUnload(LoaderHarness* const harness) {
  harness->UnloadRequested();
  harness->StartQuiescing();
  harness->DoneQuiescing();
  harness->Unload();
}

}  // namespace

TEST(LoaderHarnessTest, Init) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  EXPECT_EQ((ServableId{"test", 0}), harness.id());
  EXPECT_EQ(LoaderHarness::State::kNew, harness.state());
  EXPECT_EQ(harness.state(), harness.loader_state_snapshot<>().state);
}

TEST(LoaderHarnessTest, LoadRequested) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  TF_ASSERT_OK(harness.LoadRequested());
  EXPECT_EQ(LoaderHarness::State::kLoadRequested, harness.state());

  harness.Error(
      errors::Unknown("Transitions harness to a legally destructible state."));
}

TEST(LoaderHarnessTest, Quiesce) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  EXPECT_CALL(*loader, Unload()).WillOnce(Return());

  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  TF_ASSERT_OK(harness.UnloadRequested());
  TF_ASSERT_OK(harness.StartQuiescing());
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
      "    device: 'main' "
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
      "} ");

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(EqualsProto(available_resources)))
      .WillOnce(Return(Status::OK()));
  {
    std::unique_ptr<Thread> test_thread(Env::Default()->StartThread(
        ThreadOptions(), "test", [&available_resources, &harness]() {
          TF_ASSERT_OK(harness.LoadRequested());
          TF_ASSERT_OK(harness.LoadApproved());
          EXPECT_TRUE(harness.Load(available_resources).ok());
        }));
    // Deleting the thread here forces join and ensures that
    // LoaderHarness::Load() returns.
  }
  EXPECT_EQ(LoaderHarness::State::kReady, harness.state());

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, Unload) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  Notification unload_called;
  Notification unload_should_return;
  EXPECT_CALL(*loader, Unload()).WillOnce(Return());
  {
    std::unique_ptr<Thread> test_thread(Env::Default()->StartThread(
        ThreadOptions(), "test", [&harness]() { QuiesceAndUnload(&harness); }));
    // Deleting the thread here forces join and ensures that
    // LoaderHarness::Unload() returns.
  }
  EXPECT_EQ(LoaderHarness::State::kDisabled, harness.state());
}

TEST(LoaderHarnessTest, UnloadRequested) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  TF_ASSERT_OK(harness.UnloadRequested());
  EXPECT_EQ(LoaderHarness::State::kUnloadRequested, harness.state());

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, LoadApproved) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  EXPECT_EQ(LoaderHarness::State::kLoadApproved, harness.state());

  harness.Error(
      errors::Unknown("Transitions harness to a legally destructible state."));
}

TEST(LoaderHarnessTest, LoadError) {
  test_util::MockLoader* loader = new StrictMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(Return(errors::Unknown("test load error")));
  {
    std::unique_ptr<Thread> test_thread(
        Env::Default()->StartThread(ThreadOptions(), "test", [&harness]() {
          TF_ASSERT_OK(harness.LoadRequested());
          TF_ASSERT_OK(harness.LoadApproved());
          Status status = harness.Load(ResourceAllocation());
          EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
        }));
  }
  EXPECT_EQ(LoaderHarness::State::kError, harness.state());
}

TEST(LoaderHarnessTest, ExternallySignalledError) {
  LoaderHarness harness(ServableId{"test", 0}, nullptr);
  EXPECT_EQ(LoaderHarness::State::kNew, harness.state());
  const Status status = Status(error::UNKNOWN, "Some unknown error");
  harness.Error(status);
  EXPECT_EQ(LoaderHarness::State::kError, harness.state());
  EXPECT_EQ(status, harness.status());
}

TEST(LoaderHarnessTest, AdditionalState) {
  std::unique_ptr<int> object(new int(10));
  LoaderHarness harness({"test", 42}, nullptr, std::move(object));

  EXPECT_EQ(10, *harness.loader_state_snapshot<int>().additional_state);
  EXPECT_EQ(10, *harness.additional_state<int>());
  EXPECT_EQ(nullptr, harness.additional_state<float>());
}

TEST(LoaderHarnessTest, NoAdditionalState) {
  LoaderHarness harness({"test", 42}, nullptr);

  // Will return nullptr when there is no metadata set.
  EXPECT_FALSE(harness.loader_state_snapshot<int>().additional_state);
  EXPECT_EQ(nullptr, harness.additional_state<int>());
  EXPECT_EQ(nullptr, harness.additional_state<float>());
}

TEST(LoaderHarnessTest, NonApprovedLoadFails) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  EXPECT_FALSE(harness.Load(ResourceAllocation()).ok());
}

TEST(LoaderHarnessTest, MultipleLoadApprovedOnlyFirstOneSucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  const Status second_approve_for_loading_status = harness.LoadApproved();
  EXPECT_FALSE(second_approve_for_loading_status.ok());
  EXPECT_EQ(error::FAILED_PRECONDITION,
            second_approve_for_loading_status.code());
  EXPECT_THAT(second_approve_for_loading_status.error_message(),
              HasSubstr("cannot be approved for loading"));

  TF_ASSERT_OK(harness.Load(ResourceAllocation()));
  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, MultipleLoadsOnlyFirstOneSucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  EXPECT_CALL(*loader, Load(_)).WillRepeatedly(Return(Status::OK()));

  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));
  const Status second_load_status = harness.Load(ResourceAllocation());
  EXPECT_FALSE(second_load_status.ok());
  EXPECT_EQ(error::FAILED_PRECONDITION, second_load_status.code());
  EXPECT_THAT(second_load_status.error_message(),
              HasSubstr("cannot be loaded"));

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, MultipleUnloadRequestedOnlyFirstOneSucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  TF_ASSERT_OK(harness.LoadRequested());
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  TF_ASSERT_OK(harness.UnloadRequested());
  const Status second_status = harness.UnloadRequested();
  EXPECT_FALSE(second_status.ok());
  EXPECT_EQ(error::FAILED_PRECONDITION, second_status.code());
  EXPECT_THAT(second_status.error_message(),
              HasSubstr("cannot be transitioned to unload-requested"));

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, MultipleStartQuiescingOnlyFirstOneSucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader));

  TF_ASSERT_OK(harness.LoadRequested());
  EXPECT_CALL(*loader, Load(_)).WillOnce(Return(Status::OK()));
  TF_ASSERT_OK(harness.LoadApproved());
  TF_ASSERT_OK(harness.Load(ResourceAllocation()));

  TF_ASSERT_OK(harness.UnloadRequested());
  TF_ASSERT_OK(harness.StartQuiescing());
  const Status second_status = harness.StartQuiescing();
  EXPECT_FALSE(second_status.ok());
  EXPECT_EQ(error::FAILED_PRECONDITION, second_status.code());
  EXPECT_THAT(second_status.error_message(), HasSubstr("cannot be quiesced"));

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, RetryOnLoadErrorFinallySucceeds) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness::Options options;
  options.max_num_load_retries = 1;
  options.load_retry_interval_micros = 1;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        options);

  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs(
          []() { return errors::Unknown("test load error"); }))
      .WillOnce(InvokeWithoutArgs([]() { return Status::OK(); }));
  TF_EXPECT_OK(harness.LoadRequested());
  TF_EXPECT_OK(harness.LoadApproved());
  TF_EXPECT_OK(harness.Load(ResourceAllocation()));

  QuiesceAndUnload(&harness);
}

TEST(LoaderHarnessTest, RetryOnLoadErrorFinallyFails) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness::Options options;
  options.max_num_load_retries = 1;
  options.load_retry_interval_micros = 0;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        options);

  EXPECT_CALL(*loader, Load(_))
      .Times(2)
      .WillRepeatedly(InvokeWithoutArgs(
          []() { return errors::Unknown("test load error"); }));
  TF_ASSERT_OK(harness.LoadRequested());
  TF_ASSERT_OK(harness.LoadApproved());
  const Status status = harness.Load(ResourceAllocation());
  EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
}

// Tests cancelling load retries.
TEST(LoaderHarnessTest, RetryOnLoadErrorCancelledLoad) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness::Options options;
  options.max_num_load_retries = 10;
  options.load_retry_interval_micros = 0;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        options);

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs([&load_called, &load_should_return]() {
        return errors::Unknown("test load error");
      }))
      // If the load is called again, we return Status::OK() to fail the test.
      .WillRepeatedly(InvokeWithoutArgs([]() { return Status::OK(); }));
  std::unique_ptr<Thread> test_thread(
      Env::Default()->StartThread(ThreadOptions(), "test", [&harness]() {
        TF_ASSERT_OK(harness.LoadRequested());
        TF_ASSERT_OK(harness.LoadApproved());
        harness.set_cancel_load_retry(true);
        const Status status = harness.Load(ResourceAllocation());
        EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
      }));
}

TEST(LoaderHarnessTest, LoadAfterCancelledLoad) {
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  LoaderHarness::Options options;
  options.max_num_load_retries = 10;
  options.load_retry_interval_micros = 0;
  LoaderHarness harness(ServableId{"test", 0}, std::unique_ptr<Loader>(loader),
                        options);

  Notification load_called;
  Notification load_should_return;
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(InvokeWithoutArgs([&load_called, &load_should_return]() {
        load_called.Notify();
        load_should_return.WaitForNotification();
        return errors::Unknown("test load error");
      }))
      .WillRepeatedly(InvokeWithoutArgs([]() { return Status::OK(); }));
  {
    std::unique_ptr<Thread> test_thread(
        Env::Default()->StartThread(ThreadOptions(), "test", [&harness]() {
          TF_ASSERT_OK(harness.LoadRequested());
          TF_ASSERT_OK(harness.LoadApproved());
          const Status status = harness.Load(ResourceAllocation());
          EXPECT_THAT(status.error_message(), HasSubstr("test load error"));
        }));
    load_called.WaitForNotification();
    harness.set_cancel_load_retry(true);
    load_should_return.Notify();
  }

  const Status second_load_status = harness.Load(ResourceAllocation());
  ASSERT_FALSE(second_load_status.ok());
  EXPECT_EQ(error::FAILED_PRECONDITION, second_load_status.code());
}

}  // namespace serving
}  // namespace tensorflow
